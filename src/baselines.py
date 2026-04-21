import torch


# Sample from the empirical product of marginals by resampling each coordinate independently.
def sample_product_of_marginals(real_samples, n_samples, seed=0):
    if real_samples.ndim != 2:
        raise ValueError(f"Expected real_samples to have shape (n, d), got {tuple(real_samples.shape)}")
    if n_samples < 1:
        raise ValueError(f"Expected n_samples to be positive, got {n_samples}")

    n_real, data_dim = real_samples.shape
    generator = torch.Generator(device=real_samples.device).manual_seed(seed)
    samples = torch.empty(n_samples, data_dim, dtype=real_samples.dtype, device=real_samples.device)

    for coordinate in range(data_dim):
        indices = torch.randint(0, n_real, (n_samples,), generator=generator, device=real_samples.device)
        samples[:, coordinate] = real_samples[indices, coordinate]

    return samples


def _matrix_square_root_trace(matrix):
    symmetric_matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = torch.linalg.eigvalsh(symmetric_matrix)
    return torch.sqrt(torch.clamp(eigenvalues, min=0.0)).sum()


# Evaluate the Gaussian W2^2 objective for a diagonal covariance candidate.
def diagonal_gaussian_w2_squared(covariance, diag_vars):
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"Expected covariance to be square, got {tuple(covariance.shape)}")
    if diag_vars.ndim != 1 or diag_vars.shape[0] != covariance.shape[0]:
        raise ValueError(
            f"Expected diag_vars to have shape ({covariance.shape[0]},), got {tuple(diag_vars.shape)}"
        )

    diag_vars = torch.clamp(diag_vars, min=1e-12)
    diag_sqrt = torch.sqrt(diag_vars)
    middle = diag_sqrt[:, None] * covariance * diag_sqrt[None, :]
    return torch.trace(covariance) + diag_vars.sum() - 2 * _matrix_square_root_trace(middle)


# Fit the diagonal Gaussian that minimizes Gaussian W2 distance to the empirical Gaussian fit.
def fit_best_diagonal_gaussian_w2(real_samples, num_steps=1000, lr=0.05):
    if real_samples.ndim != 2:
        raise ValueError(f"Expected real_samples to have shape (n, d), got {tuple(real_samples.shape)}")
    if num_steps < 1:
        raise ValueError(f"Expected num_steps to be positive, got {num_steps}")
    if lr <= 0:
        raise ValueError(f"Expected lr to be positive, got {lr}")

    samples = real_samples.detach().to(dtype=torch.float64)
    mean = samples.mean(dim=0)
    covariance = torch.cov(samples.T)

    log_diag_vars = torch.log(torch.clamp(torch.diag(covariance), min=1e-6)).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([log_diag_vars], lr=lr)

    for _ in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        diag_vars = torch.exp(log_diag_vars)
        loss = diagonal_gaussian_w2_squared(covariance, diag_vars)
        loss.backward()
        optimizer.step()

    fitted_diag_vars = torch.exp(log_diag_vars).detach().to(dtype=real_samples.dtype, device=real_samples.device)
    fitted_mean = mean.detach().to(dtype=real_samples.dtype, device=real_samples.device)
    fitted_w2_squared = diagonal_gaussian_w2_squared(
        covariance, fitted_diag_vars.to(dtype=torch.float64, device=covariance.device)
    ).item()

    return {
        "mean": fitted_mean,
        "diag_vars": fitted_diag_vars,
        "w2_squared": fitted_w2_squared,
    }


# Sample from a diagonal Gaussian with the supplied mean and coordinate variances.
def sample_diagonal_gaussian(mean, diag_vars, n_samples, seed=0):
    if mean.ndim != 1:
        raise ValueError(f"Expected mean to have shape (d,), got {tuple(mean.shape)}")
    if diag_vars.ndim != 1 or diag_vars.shape != mean.shape:
        raise ValueError(f"Expected diag_vars to have shape {tuple(mean.shape)}, got {tuple(diag_vars.shape)}")
    if n_samples < 1:
        raise ValueError(f"Expected n_samples to be positive, got {n_samples}")

    std = torch.sqrt(torch.clamp(diag_vars, min=1e-12))
    generator = torch.Generator(device=mean.device).manual_seed(seed)
    noise = torch.randn(n_samples, mean.shape[0], generator=generator, device=mean.device, dtype=mean.dtype)
    return mean + noise * std
