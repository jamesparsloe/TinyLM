import torch
from tinylm.tokenizers import Utf8Tokenizer
from tinylm.train import seed_all
import click
from tinylm import viz
import os


@click.command()
@click.option("--checkpoint-path", default="./runs/4ccbskk7/TinyLM-010000.pt")
@click.option("--prompt", default="A long time ago ")
@click.option("--top-k", type=int, default=256)
@click.option("--temperature", type=float, default=1.0)
@click.option("--seed", type=int, default=42)
@click.option("--max-seqlen", type=int)
def main(
    checkpoint_path: str,
    prompt: str,
    top_k: int | None,
    temperature: int,
    seed: int,
    max_seqlen: int | None,
):
    seed_all(seed)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    step = checkpoint["step"]

    config = checkpoint["config"]
    model_config = config["model"]

    # use PyTorch impl to materialize the attention weights
    model_config["use_flash_attn"] = False

    kind = model_config["kind"]

    if kind == "gpt":
        from tinylm.models import GPTConfig, GPT

        model_config = GPTConfig(**model_config)
        model = GPT(model_config).eval().cuda()
        _ = model.load_state_dict(checkpoint["model"])
    else:
        raise ValueError(f'Model kind "{kind}" is not currently supported.')

    n_parameters = sum(p.numel() for p in model.parameters())

    print(f"{kind} {step=} {n_parameters/1e6:.2f}M {temperature=} {top_k=} {prompt=}")
    print()

    tokenizer = Utf8Tokenizer()
    input_ids = tokenizer.encode([prompt])
    input_ids = torch.tensor(input_ids, device="cuda")
    input_ids = input_ids[:, :-1]
    # input_ids = torch.tensor([[tokenizer.bos_token_id]], device="cuda")

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        out = model.generate(
            input_ids,
            max_seqlen=max_seqlen or model_config.max_seqlen,
            top_k=top_k,
            temperature=temperature,
            need_weights=True,
        )

    token_ids = out["token_ids"]
    attn_weights = out["attn_weights"]

    run_id = checkpoint_path.split("/")[-2]
    os.makedirs("./viz", exist_ok=True)
    viz_path = f"./viz/{run_id}-{kind}-{step:06d}-attn.png"
    viz.plot_attn(
        attn_weights,
        viz_path,
    )

    generated = tokenizer.decode(token_ids.cpu().tolist())

    print(generated[0])


if __name__ == "__main__":
    main()
