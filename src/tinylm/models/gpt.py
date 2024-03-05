import torch
from torch import Tensor
import torch.nn as nn
from pydantic import BaseModel
from einops import rearrange
import torch.nn.functional as F
from typing import Literal
import math


class GPTConfig(BaseModel):
    kind: Literal["gpt"]
    vocab_size: int = 256 + 3
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    bias: bool = False
    dropout: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    max_seqlen: int = 4096
    pad_vocab_size_multiple: int = 8
    amp_dtype: str = "bfloat16"


class MHA(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        bias: bool,
        dropout: float,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout

        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)

        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=bias), nn.Dropout(dropout)
        )

    def forward(self, x: Tensor):
        qkv = self.Wqkv(x)

        qkv = rearrange(qkv, "B T (three h d) -> B three h T d", three=3, d=self.d_head)
        q, k, v = qkv.unbind(dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )

        out = self.out_proj(rearrange(out, "... h T d -> ... T (h d)"))

        return out


class RMSNorm(torch.nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int | None = None,
        activation=F.gelu,
        bias: bool = False,
    ):
        super().__init__()
        d_hidden = d_hidden or 4 * d_model
        self.fc1 = nn.Linear(d_model, d_hidden, bias=bias)
        self.activation = activation
        self.fc2 = nn.Linear(d_hidden, d_model, bias=bias)

    def forward(self, x: Tensor):
        return self.fc2(self.activation(self.fc1(x)))


class Block(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        bias: bool,
        dropout: float,
    ):
        super().__init__()

        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MHA(
            d_model=d_model,
            n_heads=n_heads,
            bias=bias,
            dropout=dropout,
        )

        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: Tensor):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_layers: int,
        bias: bool,
        dropout: float,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model=d_model,
                    n_heads=n_heads,
                    bias=bias,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor):
        for block in self.blocks:
            x = block(x)

        return self.norm(x)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        vocab_size = config.vocab_size

        if vocab_size % config.pad_vocab_size_multiple != 0:
            vocab_size += config.pad_vocab_size_multiple - (
                vocab_size % config.pad_vocab_size_multiple
            )

        self.emb = nn.Embedding(
            vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.pos_emb = nn.Embedding(config.max_seqlen, config.d_model)
        self.decoder = Decoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            bias=config.bias,
            dropout=config.dropout,
        )

        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=config.bias)

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        target_ids: Tensor | None = None,
    ):
        device = input_ids.device
        B, T = input_ids.size()

        emb = self.emb(input_ids)

        if position_ids is None:
            position_ids = torch.arange(T, device=device)

        pos_emb = self.pos_emb(position_ids)

        emb = emb + pos_emb

        out = self.decoder(emb)

        logits = self.lm_head(out)

        if target_ids is None:
            return logits
        else:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=self.emb.padding_idx,
            )
            return loss

    @torch.inference_mode()
    def generate(self, input_ids: Tensor, max_seqlen: int, top_k: int | None):
        device = input_ids.device
        B, T = input_ids.size()

        while T < max_seqlen:
            logits = self(input_ids)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            else:
                next_token = logits[:, -1].argmax(-1, keepdim=True)

            input_ids = torch.cat((input_ids, next_token), dim=-1)
            T += 1

        return input_ids


if __name__ == "__main__":
    from tinylm.tokenizers import Utf8Tokenizer

    tokenizer = Utf8Tokenizer()

    texts = ["hello", "world", "Does this work?"]
    token_ids = tokenizer.encode(texts)

    device = "cuda:0"

    input_ids = torch.tensor(token_ids, device=device)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=4,
        bias=True,
        dropout=0.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = GPT(config).to(device)

    enabled = config.amp_dtype == "bfloat16"
    dtype = torch.bfloat16 if enabled else torch.float32

    with torch.cuda.amp.autocast(dtype=dtype, enabled=enabled):
        input_ids, target_ids = input_ids[..., :-1], input_ids[..., 1:].contiguous()
        loss = model(input_ids, target_ids=target_ids)