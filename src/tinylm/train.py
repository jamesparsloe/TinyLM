import functools
import math
import os
import random
import time

import click
import numpy as np
import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader, Dataset

import wandb
from tinylm.models import GPT, GPTConfig

from .tokenizers import Utf8Tokenizer

ModelConfig = GPTConfig


def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def warmup_then_cosine_decay(
    step: int, *, warmup_steps: int, steps: int, min_lr: float, max_lr: float
):
    if step < warmup_steps:
        return min_lr + step * (max_lr - min_lr) / (warmup_steps)
    elif step > steps:
        return min_lr
    else:
        decay_ratio = (step - warmup_steps) / (steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


def build_optimizer(
    model: nn.Module, *, weight_decay: float, lr: float, betas: tuple[float, float]
):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, fused=True)

    return optimizer


class TrainConfig(BaseModel):
    seed: int = 42
    batch_size: int = 256
    micro_batch_size: int = 32
    total_steps: int = 10_000
    warmup_steps: int = 100
    start_lr: float = 3e-5
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    log_every: int = 10
    checkpoint_every: int = 500

    compile: bool = True

    @property
    def gradient_accumulation_steps(self):
        return self.batch_size // self.micro_batch_size


class Config(BaseModel):
    train: TrainConfig = TrainConfig()
    model: ModelConfig = Field(discriminator="kind")


class TinyLMDataset(Dataset):
    def __init__(self, ds, max_seqlen: int):
        self.tokenizer = Utf8Tokenizer()
        self.max_seqlen = max_seqlen
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        text = item["text"]
        token_ids = self.tokenizer.encode(text)
        token_ids = token_ids[: self.max_seqlen]
        return token_ids


def collate(batch, *, pad_token_id: int):
    max_len = max(len(t) for t in batch)
    padded = [t + [pad_token_id] * (max_len - len(t)) for t in batch]
    return torch.tensor(padded)


def cycle(dl):
    while True:
        for item in dl:
            yield item


def decompile_state_dict(state_dict):
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.command("--edit", is_flag=True)
def main(config_path: str, edit: bool):
    name = "TinyLM"

    with open(config_path, "r") as f:
        s = f.read()

    if edit:
        s = click.edit(s)

    config = Config(**yaml.safe_load(s))

    device = "cuda"

    train_config = config.train
    model_config = config.model

    seed_all(train_config.seed)

    model = GPT(model_config).to(device)

    if train_config.compile:
        model = torch.compile(model)

    optimizer = build_optimizer(
        model,
        weight_decay=train_config.weight_decay,
        lr=train_config.start_lr,
        betas=train_config.betas,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{n_params / 1e6:.2f}M parameters")

    ds = load_dataset("roneneldan/TinyStories")
    train_ds, val_ds = (
        TinyLMDataset(ds["train"], max_seqlen=model_config.max_seqlen),
        TinyLMDataset(ds["validation"], max_seqlen=model_config.max_seqlen),
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=train_config.micro_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=functools.partial(collate, pad_token_id=model_config.pad_token_id),
    )

    train_dl_iter = cycle(train_dl)

    amp_enabled = model_config.amp_dtype == "bfloat16"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    step = 0

    def next_batch():
        input_ids = next(train_dl_iter)
        input_ids, target_ids = input_ids[:, :-1], input_ids[:, 1:].contiguous()
        input_ids, target_ids = (
            input_ids.to(device, non_blocking=True),
            target_ids.to(device, non_blocking=True),
        )
        return input_ids, target_ids

    get_lr = functools.partial(
        warmup_then_cosine_decay,
        steps=train_config.total_steps,
        min_lr=train_config.start_lr,
        max_lr=train_config.lr,
        warmup_steps=train_config.warmup_steps,
    )

    input_ids, target_ids = next_batch()

    run = wandb.init(project=name, config=config.model_dump())

    run_dir = os.path.join("./runs", run.id)
    os.makedirs(run_dir, exist_ok=True)

    t1 = time.perf_counter()

    while step < train_config.total_steps:
        lr = get_lr(step)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        for micro_step in range(train_config.gradient_accumulation_steps):
            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=amp_enabled):
                loss = model(input_ids, target_ids=target_ids)
                loss = loss / train_config.gradient_accumulation_steps
                loss.backward()

            input_ids, target_ids = next_batch()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.grad_clip
        )

        optimizer.step()
        optimizer.zero_grad()
        step += 1

        if step % train_config.log_every == 0:
            t2 = time.perf_counter()
            throughput = train_config.log_every * train_config.batch_size / (t2 - t1)
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/throughput": throughput,
                    "train/lr": lr,
                },
                step=step,
            )
            t1 = t2

        if step % train_config.checkpoint_every == 0:
            state_dict = (
                decompile_state_dict(model.state_dict())
                if train_config.compile
                else model.state_dict()
            )

            checkpoint = {
                "config": config.model_dump(),
                "model": state_dict,
                "optimizer": optimizer.state_dict(),
                "step": step,
            }

            checkpoint_path = os.path.join(run_dir, f"{name}-{step:06d}.pt")

            torch.save(checkpoint, checkpoint_path)

            print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
