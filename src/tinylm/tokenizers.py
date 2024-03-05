from typing import Literal


class Utf8Tokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.n_special_tokens = 3
        self.vocab_size = 256 + self.n_special_tokens

    def encode(
        self,
        texts: list[str] | str,
        add_special_tokens: bool = True,
        padding: Literal["longest", "fixed", "none"] = "longest",
        fixed_seqlen: int = 2048,
    ) -> list[list[int]]:
        batched = isinstance(texts, list)

        if not batched:
            texts = [texts]

        pre = [self.bos_token_id] if add_special_tokens else []
        suf = [self.eos_token_id] if add_special_tokens else []

        tokens = [
            (pre + [t + self.n_special_tokens for t in text.encode("utf-8")] + suf)
            for text in texts
        ]

        if padding == "longest":
            max_seqlen = max(len(t) for t in tokens)
            tokens = [t + [self.pad_token_id] * (max_seqlen - len(t)) for t in tokens]
        elif padding == "fixed":
            tokens = [t[:fixed_seqlen] for t in tokens]
            tokens = [t + [self.pad_token_id] * (fixed_seqlen - len(t)) for t in tokens]

        if not batched:
            tokens = tokens[0]

        return tokens

    def decode(
        self, tokens: list[list[int]], remove_special_tokens: bool = True
    ) -> list[str]:
        return [
            "".join(
                [
                    chr(t - self.n_special_tokens)
                    for t in token
                    if t >= self.n_special_tokens and remove_special_tokens
                ]
            )
            for token in tokens
        ]


if __name__ == "__main__":
    tokenizer = Utf8Tokenizer()

    texts = ["hello", "world", "Does this work?"]

    tokens = tokenizer.encode(texts, padding="fixed")

    print(tokens)

    decoded = tokenizer.decode(tokens)

    print(decoded)
