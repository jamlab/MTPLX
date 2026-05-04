from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from mtplx.draft_lm_head import _install_draft_lm_head


class _TiedEmbeddingText(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        embedding = nn.Embedding(32, 32)
        embedding.weight = mx.ones((32, 32), dtype=mx.bfloat16)
        self.model = SimpleNamespace(
            embed_tokens=nn.QuantizedEmbedding.from_embedding(
                embedding,
                group_size=32,
                bits=4,
                mode="affine",
            )
        )
        self.args = SimpleNamespace(tie_word_embeddings=True)


def test_install_draft_lm_head_supports_tied_quantized_embeddings() -> None:
    rt = SimpleNamespace(model=_TiedEmbeddingText())

    report = _install_draft_lm_head(rt, bits=4, group_size=32, mode="affine")
    head = rt.model._mtplx_draft_lm_head
    logits = head(mx.ones((1, 2, 32), dtype=mx.bfloat16))
    mx.eval(logits)

    assert report["source"] == "tied_embedding"
    assert report["reused_existing_quantization"] is True
    assert logits.shape == (1, 2, 32)
