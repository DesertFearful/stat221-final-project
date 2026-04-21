import math

import torch


# Take a reproducible subset so exact OT stays tractable during sweeps.
def subsample_samples(samples, max_samples, seed):
    subset = samples
    if max_samples is not None and samples.shape[0] > max_samples:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(samples.shape[0], generator=generator)[:max_samples]
        subset = samples[indices]

    return subset.detach().cpu().to(dtype=torch.float64)


# Compute exact empirical W1 and W2 on a fixed subsample using POT.
def estimate_pot_wasserstein(real_samples, fake_samples, max_samples=512, seed=0):
    try:
        import ot
    except ImportError as exc:
        raise ImportError("POT is required for Wasserstein evaluation. Install it with `pip install POT`.") from exc

    real_subset = subsample_samples(real_samples, max_samples, seed).numpy()
    fake_subset = subsample_samples(fake_samples, max_samples, seed + 1).numpy()

    a = ot.unif(real_subset.shape[0])
    b = ot.unif(fake_subset.shape[0])

    cost_w1 = ot.dist(real_subset, fake_subset, metric="euclidean")
    w1 = ot.emd2(a, b, cost_w1)

    cost_w2 = ot.dist(real_subset, fake_subset)
    w2_squared = ot.emd2(a, b, cost_w2)
    w2 = math.sqrt(max(w2_squared, 0.0))

    return {
        "w1": float(w1),
        "w2": float(w2),
    }
