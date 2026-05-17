import math

import mlx.core as mx
import numpy as np

from mtplx.fast_sampling import (
    batched_sparse_distributions_from_mlx_logits,
    sparse_distribution_from_mlx_logits,
    sparse_distributions_from_mlx_logits,
)
from mtplx.sampling import SamplerConfig


def test_sparse_distribution_nan_mass_falls_back_to_one_hot():
    dist = sparse_distribution_from_mlx_logits(
        mx.array([math.nan, math.nan], dtype=mx.float32),
        SamplerConfig(temperature=0.6, top_p=0.95, top_k=2),
    )

    assert dist is not None
    assert np.isfinite(dist.probs).all()
    assert dist.probs.sum() == 1.0


def test_sparse_distribution_batch_nan_mass_falls_back_to_one_hot():
    dists = sparse_distributions_from_mlx_logits(
        mx.array([[math.nan, math.nan], [1.0, 0.0]], dtype=mx.float32),
        SamplerConfig(temperature=0.6, top_p=0.95, top_k=2),
    )

    assert dists is not None
    assert len(dists) == 2
    assert np.isfinite(dists[0].probs).all()
    assert dists[0].probs.sum() == 1.0


def test_batched_sparse_distribution_nan_mass_falls_back_to_one_hot():
    batch = batched_sparse_distributions_from_mlx_logits(
        mx.array([[math.nan, math.nan], [1.0, 0.0]], dtype=mx.float32),
        SamplerConfig(temperature=0.6, top_p=0.95, top_k=2),
    )

    assert batch is not None
    assert np.isfinite(batch.probs).all()
    assert np.allclose(batch.probs.sum(axis=1), 1.0)
