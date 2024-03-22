import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pydantic import BaseModel
from torch import Tensor


class GPTConfig(BaseModel):
    kind: Literal["gpt"]
    tokenizer: str = "utf8"
    vocab_size: int = 256 + 3
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    bias: bool = False
    dropout: float = 0.1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    max_seqlen: int = 2048
    pad_vocab_size_multiple: int = 8
    amp_dtype: str = "bfloat16"
    use_flash_attn: bool = True


class MHA(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        bias: bool,
        dropout: float,
        use_flash_attn: bool = True,
        max_seqlen: int = 2048,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.use_flash_attn = use_flash_attn
        self.max_seqlen = max_seqlen

        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=bias), nn.Dropout(dropout)
        )

        if not self.use_flash_attn:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(max_seqlen, max_seqlen)).view(
                    1, 1, max_seqlen, max_seqlen
                ),
                persistent=False,
            )

    def forward(self, x: Tensor):
        B, T, d = x.size()
        qkv = self.Wqkv(x)

        qkv = rearrange(qkv, "B T (three h d) -> B three h T d", three=3, d=self.d_head)
        q, k, v = qkv.unbind(dim=1)

        attn_weights = None

        if self.use_flash_attn:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_weights = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
            attn_weights = attn_weights.masked_fill(
                self.causal_mask[:, :, :T, :T] == 0, -float("inf")
            )
            attn_weights = F.softmax(attn_weights, dim=-1)
            out = attn_weights @ v

        out = self.out_proj(rearrange(out, "... h T d -> ... T (h d)"))

        return out, attn_weights


class RMSNorm(nn.Module):
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
        use_flash_attn: bool,
    ):
        super().__init__()

        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MHA(
            d_model=d_model,
            n_heads=n_heads,
            bias=bias,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
        )

        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: Tensor):
        attn_out, attn_weights = self.attn(self.attn_norm(x))

        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))

        return x, attn_weights


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_layers: int,
        bias: bool,
        dropout: float,
        use_flash_attn: bool,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model=d_model,
                    n_heads=n_heads,
                    bias=bias,
                    dropout=dropout,
                    use_flash_attn=use_flash_attn,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, need_weights: bool = False):
        all_attn_weights = []
        for block in self.blocks:
            x, attn_weights = block(x)
            all_attn_weights.append(attn_weights)

        if need_weights:
            all_attn_weights = torch.concat(all_attn_weights, dim=0)
        else:
            all_attn_weights = None

        return self.norm(x), all_attn_weights


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        vocab_size = config.vocab_size

        if vocab_size % config.pad_vocab_size_multiple != 0:
            vocab_size += config.pad_vocab_size_multiple - (
                vocab_size % config.pad_vocab_size_multiple
            )

        self.emb = nn.Sequential(
            nn.Embedding(vocab_size, config.d_model, padding_idx=config.pad_token_id),
            nn.Dropout(config.dropout),
        )
        self.pos_emb = nn.Embedding(config.max_seqlen, config.d_model)
        self.decoder = Decoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            bias=config.bias,
            dropout=config.dropout,
            use_flash_attn=config.use_flash_attn,
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
        need_weights: bool = False,
    ):
        device = input_ids.device
        B, T = input_ids.size()

        emb = self.emb(input_ids)

        if position_ids is None:
            position_ids = torch.arange(T, device=device)

        pos_emb = self.pos_emb(position_ids)

        emb = emb + pos_emb

        out, attn_weights = self.decoder(emb, need_weights=need_weights)

        logits = self.lm_head(out)

        if target_ids is None:
            return {
                "logits": logits,
                "attn_weights": attn_weights,
            }
        else:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=self.config.pad_token_id,
            )
            return loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        max_seqlen: int,
        top_k: int | None = None,
        temperature: float = 1.0,
        need_weights: bool = False,
    ):
        device = input_ids.device
        B, T = input_ids.size()

        attn_weights = None

        while T < max_seqlen:
            out = self(input_ids, need_weights=need_weights)
            logits = out["logits"]
            attn_weights = out["attn_weights"]

            logits = logits[:, -1]

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(-1, keepdim=True)

            if next_token.item() == self.config.eos_token_id:
                break

            input_ids = torch.cat((input_ids, next_token), dim=-1)
            T += 1

        return {
            "token_ids": input_ids,
            "attn_weights": attn_weights,
        }


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
