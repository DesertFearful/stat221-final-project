import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from experiment_2d_gaussian import (
    DEFAULT_CRITIC_FEATURE_MAP,
    DEFAULT_GP_LAMBDA,
    DEFAULT_N_CRITIC,
    DEFAULT_NUM_EPOCHS,
    build_hidden_dims,
    make_dataloader,
    make_equicorrelated_gaussian_samples,
    resolve_device,
    resolve_learning_rates,
    save_loss_plot,
)
from src.baselines import fit_best_diagonal_gaussian_w2, sample_diagonal_gaussian, sample_product_of_marginals
from src.model import Critic, FactorizedGenerator
from src.ot_metrics import estimate_pot_wasserstein
from src.train_wgan import WGANTrainer


METHOD_ORDER = ["Best diagonal Gaussian", "Best W1 diagonal Gaussian", "Product marginals", "WGAN"]
METHOD_COLORS = {
    "Best diagonal Gaussian": "tab:green",
    "Best W1 diagonal Gaussian": "tab:purple",
    "Product marginals": "tab:orange",
    "WGAN": "tab:blue",
}


def rho_tag(rho):
    rho_string = f"{rho:.3f}".rstrip("0").rstrip(".")
    if not rho_string:
        rho_string = "0"
    return rho_string.replace("-", "m").replace(".", "p")


def theoretical_equicorrelated_diagonal_variance(data_dim, rho):
    average_sqrt_eigenvalue = (
        math.sqrt(1.0 + (data_dim - 1) * rho) + (data_dim - 1) * math.sqrt(1.0 - rho)
    ) / data_dim
    return average_sqrt_eigenvalue * average_sqrt_eigenvalue


def covariance_to_correlation(covariance):
    variances = torch.clamp(torch.diag(covariance), min=1e-12)
    scale = torch.sqrt(variances)
    correlation = covariance / (scale[:, None] * scale[None, :])
    return torch.clamp(correlation, min=-1.0, max=1.0)


def off_diagonal_mean_abs(matrix):
    if matrix.shape[0] < 2:
        return 0.0

    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return matrix[mask].abs().mean().item()


def fit_best_scalar_diagonal_gaussian_w1(
    train_samples,
    validation_samples,
    min_var,
    max_var,
    grid_size,
    refine_rounds,
    ot_eval_samples,
    seed,
):
    if min_var <= 0:
        raise ValueError(f"Expected min_var to be positive, got {min_var}")
    if max_var <= min_var:
        raise ValueError(f"Expected max_var > min_var, got min_var={min_var}, max_var={max_var}")
    if grid_size < 2:
        raise ValueError(f"Expected grid_size to be at least 2, got {grid_size}")
    if refine_rounds < 1:
        raise ValueError(f"Expected refine_rounds to be positive, got {refine_rounds}")

    mean = train_samples.mean(dim=0)
    generator = torch.Generator(device=validation_samples.device).manual_seed(seed)
    base_noise = torch.randn(
        validation_samples.shape[0],
        validation_samples.shape[1],
        generator=generator,
        device=validation_samples.device,
        dtype=validation_samples.dtype,
    )
    lower = min_var
    upper = max_var
    best_var = None
    best_w1 = float("inf")

    for _ in range(refine_rounds):
        grid = torch.linspace(lower, upper, grid_size, dtype=validation_samples.dtype)
        round_values = []

        for var_value in grid.tolist():
            candidate_samples = mean + math.sqrt(var_value) * base_noise
            metrics = estimate_pot_wasserstein(
                validation_samples,
                candidate_samples,
                max_samples=ot_eval_samples,
                seed=seed,
            )
            round_values.append((var_value, metrics["w1"]))
            if metrics["w1"] < best_w1:
                best_w1 = metrics["w1"]
                best_var = var_value

        round_best_index = min(range(len(round_values)), key=lambda index: round_values[index][1])
        if round_best_index == 0:
            lower = min_var
            upper = round_values[1][0]
        elif round_best_index == len(round_values) - 1:
            lower = round_values[-2][0]
            upper = max_var
        else:
            lower = round_values[round_best_index - 1][0]
            upper = round_values[round_best_index + 1][0]

    diag_vars = torch.full_like(mean, float(best_var))
    return {
        "mean": mean,
        "diag_vars": diag_vars,
        "validation_w1": best_w1,
    }


def compute_sample_summary(real_samples, candidate_samples, ot_eval_samples, seed, theoretical_diag_var):
    real_mean = real_samples.mean(dim=0)
    candidate_mean = candidate_samples.mean(dim=0)
    real_covariance = torch.cov(real_samples.T)
    candidate_covariance = torch.cov(candidate_samples.T)
    real_diag = torch.diag(real_covariance)
    candidate_diag = torch.diag(candidate_covariance)
    candidate_diag_var_mean = candidate_diag.mean().item()
    covariance_gap = candidate_covariance - real_covariance
    candidate_correlation = covariance_to_correlation(candidate_covariance)
    ot_metrics = estimate_pot_wasserstein(real_samples, candidate_samples, max_samples=ot_eval_samples, seed=seed)

    return {
        "mean_error": torch.norm(candidate_mean - real_mean).item(),
        "diag_var_error": (candidate_diag - real_diag).abs().mean().item(),
        "real_diag_var_mean": real_diag.mean().item(),
        "candidate_diag_var_mean": candidate_diag_var_mean,
        "theoretical_diag_var": theoretical_diag_var,
        "candidate_minus_theoretical_diag_var": candidate_diag_var_mean - theoretical_diag_var,
        "offdiag_cov_error": off_diagonal_mean_abs(covariance_gap),
        "cov_fro_error": torch.linalg.norm(covariance_gap).item(),
        "avg_abs_corr": off_diagonal_mean_abs(candidate_correlation),
        "w1": ot_metrics["w1"],
        "w2": ot_metrics["w2"],
    }


def save_results_csv(results, output_path):
    if not results:
        return

    fieldnames = list(results[0].keys())
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def save_seed_summary(results, output_path):
    lines = []
    for result in results:
        lines.extend(
            [
                result["method"],
                f"  W1: {result['w1']:.4f}",
                f"  W2: {result['w2']:.4f}",
                f"  mean_error: {result['mean_error']:.4f}",
                f"  diag_var_error: {result['diag_var_error']:.4f}",
                f"  candidate_diag_var_mean: {result['candidate_diag_var_mean']:.4f}",
                f"  theoretical_diag_var: {result['theoretical_diag_var']:.4f}",
                f"  candidate_minus_theoretical_diag_var: {result['candidate_minus_theoretical_diag_var']:.4f}",
                f"  offdiag_cov_error: {result['offdiag_cov_error']:.4f}",
                f"  cov_fro_error: {result['cov_fro_error']:.4f}",
                f"  avg_abs_corr: {result['avg_abs_corr']:.4f}",
                "",
            ]
        )

    output_path.write_text("\n".join(lines).strip() + "\n")


def aggregate_method_results(per_seed_results):
    metrics = [
        "w1",
        "w2",
        "mean_error",
        "diag_var_error",
        "real_diag_var_mean",
        "candidate_diag_var_mean",
        "theoretical_diag_var",
        "candidate_minus_theoretical_diag_var",
        "offdiag_cov_error",
        "cov_fro_error",
        "avg_abs_corr",
    ]
    grouped = {}

    for result in per_seed_results:
        key = (result["data_dim"], result["rho"], result["method"])
        grouped.setdefault(key, []).append(result)

    aggregate = []
    data_dims = sorted({result["data_dim"] for result in per_seed_results})
    rhos = sorted({result["rho"] for result in per_seed_results})

    for data_dim in data_dims:
        for rho in rhos:
            for method in METHOD_ORDER:
                key = (data_dim, rho, method)
                if key not in grouped:
                    continue

                group = grouped[key]
                summary = {"data_dim": data_dim, "rho": rho, "method": method, "num_seeds": len(group)}
                for metric in metrics:
                    values = torch.tensor([item[metric] for item in group], dtype=torch.float64)
                    summary[f"{metric}_mean"] = values.mean().item()
                    summary[f"{metric}_std"] = values.std(unbiased=False).item()

                aggregate.append(summary)

    return aggregate


def aggregate_training_results(training_results):
    metrics = [
        "selected_epoch",
        "selected_eval_w1",
        "selected_eval_mean_error",
        "selected_eval_diag_var_error",
        "selected_checkpoint_score",
        "best_epoch",
        "best_eval_w1",
        "best_eval_mean_error",
        "best_eval_diag_var_error",
        "best_checkpoint_score",
    ]
    grouped = {}

    for result in training_results:
        key = (result["data_dim"], result["rho"])
        grouped.setdefault(key, []).append(result)

    aggregate = []
    for data_dim, rho in sorted(grouped):
        group = grouped[(data_dim, rho)]
        summary = {"data_dim": data_dim, "rho": rho, "num_seeds": len(group)}
        for metric in metrics:
            values = torch.tensor([item[metric] for item in group], dtype=torch.float64)
            summary[f"{metric}_mean"] = values.mean().item()
            summary[f"{metric}_std"] = values.std(unbiased=False).item()
        aggregate.append(summary)

    return aggregate


def compute_gap_results(per_seed_results):
    grouped = {}

    for result in per_seed_results:
        key = (result["data_dim"], result["rho"], result["seed"])
        grouped.setdefault(key, {})[result["method"]] = result

    gap_results = []
    for (data_dim, rho, seed), methods in sorted(grouped.items()):
        if not all(method in methods for method in METHOD_ORDER):
            continue

        diagonal = methods["Best diagonal Gaussian"]
        w1_diagonal = methods["Best W1 diagonal Gaussian"]
        product = methods["Product marginals"]
        wgan = methods["WGAN"]
        gap_results.append(
            {
                "data_dim": data_dim,
                "rho": rho,
                "seed": seed,
                "product_minus_diag_w1": product["w1"] - diagonal["w1"],
                "product_minus_diag_w2": product["w2"] - diagonal["w2"],
                "w1diag_minus_diag_w1": w1_diagonal["w1"] - diagonal["w1"],
                "w1diag_minus_diag_w2": w1_diagonal["w2"] - diagonal["w2"],
                "product_minus_w1diag_w1": product["w1"] - w1_diagonal["w1"],
                "product_minus_w1diag_w2": product["w2"] - w1_diagonal["w2"],
                "wgan_minus_diag_w1": wgan["w1"] - diagonal["w1"],
                "wgan_minus_diag_w2": wgan["w2"] - diagonal["w2"],
                "wgan_minus_w1diag_w1": wgan["w1"] - w1_diagonal["w1"],
                "wgan_minus_w1diag_w2": wgan["w2"] - w1_diagonal["w2"],
            }
        )

    return gap_results


def aggregate_gap_results(gap_results):
    metrics = [
        "product_minus_diag_w1",
        "product_minus_diag_w2",
        "w1diag_minus_diag_w1",
        "w1diag_minus_diag_w2",
        "product_minus_w1diag_w1",
        "product_minus_w1diag_w2",
        "wgan_minus_diag_w1",
        "wgan_minus_diag_w2",
        "wgan_minus_w1diag_w1",
        "wgan_minus_w1diag_w2",
    ]
    grouped = {}

    for result in gap_results:
        key = (result["data_dim"], result["rho"])
        grouped.setdefault(key, []).append(result)

    aggregate = []
    for data_dim, rho in sorted(grouped):
        group = grouped[(data_dim, rho)]
        summary = {"data_dim": data_dim, "rho": rho, "num_seeds": len(group)}
        for metric in metrics:
            values = torch.tensor([item[metric] for item in group], dtype=torch.float64)
            summary[f"{metric}_mean"] = values.mean().item()
            summary[f"{metric}_std"] = values.std(unbiased=False).item()
        aggregate.append(summary)

    return aggregate


def save_metric_grid(aggregate_results, output_path):
    metrics = [
        ("w1", "W1"),
        ("w2", "W2"),
        ("diag_var_error", "Diagonal variance error"),
        ("offdiag_cov_error", "Off-diagonal covariance error"),
    ]
    rhos = sorted({row["rho"] for row in aggregate_results})
    data_dims = sorted({row["data_dim"] for row in aggregate_results})
    figure, axes = plt.subplots(len(metrics), len(rhos), figsize=(4.5 * len(rhos), 3.5 * len(metrics)), squeeze=False)

    for row_index, (metric, label) in enumerate(metrics):
        for col_index, rho in enumerate(rhos):
            axis = axes[row_index][col_index]
            for method in METHOD_ORDER:
                method_rows = [
                    row
                    for row in aggregate_results
                    if row["rho"] == rho and row["method"] == method
                ]
                if not method_rows:
                    continue

                x_values = [row["data_dim"] for row in method_rows]
                y_values = [row[f"{metric}_mean"] for row in method_rows]
                y_errors = [row[f"{metric}_std"] for row in method_rows]
                axis.errorbar(
                    x_values,
                    y_values,
                    yerr=y_errors,
                    marker="o",
                    capsize=4,
                    label=method,
                    color=METHOD_COLORS[method],
                )

            if row_index == 0:
                axis.set_title(f"rho = {rho:.2f}")
            if col_index == 0:
                axis.set_ylabel(label)
            if row_index == len(metrics) - 1:
                axis.set_xlabel("dimension p")
            axis.set_xticks(data_dims)
            axis.grid(alpha=0.25)

    axes[0][0].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_variance_target_grid(aggregate_results, output_path):
    rhos = sorted({row["rho"] for row in aggregate_results})
    data_dims = sorted({row["data_dim"] for row in aggregate_results})
    figure, axes = plt.subplots(1, len(rhos), figsize=(5.0 * len(rhos), 4), squeeze=False, sharey=True)

    for col_index, rho in enumerate(rhos):
        axis = axes[0][col_index]
        target_values = [theoretical_equicorrelated_diagonal_variance(data_dim, rho) for data_dim in data_dims]
        axis.plot(data_dims, target_values, color="black", linestyle="--", marker="o", label="W2 diagonal target")
        axis.axhline(1.0, color="0.5", linestyle=":", label="true marginal variance")

        for method in METHOD_ORDER:
            method_rows = [
                row
                for row in aggregate_results
                if row["rho"] == rho and row["method"] == method
            ]
            if not method_rows:
                continue

            method_rows = sorted(method_rows, key=lambda row: row["data_dim"])
            x_values = [row["data_dim"] for row in method_rows]
            y_values = [row["candidate_diag_var_mean_mean"] for row in method_rows]
            y_errors = [row["candidate_diag_var_mean_std"] for row in method_rows]
            axis.errorbar(
                x_values,
                y_values,
                yerr=y_errors,
                marker="o",
                capsize=4,
                label=method,
                color=METHOD_COLORS[method],
            )

        axis.set_title(f"rho = {rho:.2f}")
        axis.set_xlabel("dimension p")
        axis.set_xticks(data_dims)
        axis.grid(alpha=0.25)
        if col_index == 0:
            axis.set_ylabel("mean coordinate variance")

    axes[0][0].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def build_gap_matrix(gap_aggregate, data_dims, rhos, metric):
    matrix = torch.full((len(rhos), len(data_dims)), float("nan"), dtype=torch.float64)
    row_lookup = {(row["rho"], row["data_dim"]): row for row in gap_aggregate}

    for rho_index, rho in enumerate(rhos):
        for dim_index, data_dim in enumerate(data_dims):
            row = row_lookup.get((rho, data_dim))
            if row is not None:
                matrix[rho_index, dim_index] = row[f"{metric}_mean"]

    return matrix


def save_gap_heatmaps(gap_aggregate, output_path):
    rhos = sorted({row["rho"] for row in gap_aggregate})
    data_dims = sorted({row["data_dim"] for row in gap_aggregate})
    specs = [
        ("product_minus_diag_w1", "Product - W2 diagonal (W1)"),
        ("w1diag_minus_diag_w1", "W1 diagonal - W2 diagonal (W1)"),
        ("wgan_minus_w1diag_w1", "WGAN - W1 diagonal (W1)"),
        ("product_minus_diag_w2", "Product - W2 diagonal (W2)"),
        ("w1diag_minus_diag_w2", "W1 diagonal - W2 diagonal (W2)"),
        ("wgan_minus_w1diag_w2", "WGAN - W1 diagonal (W2)"),
    ]
    matrices = [build_gap_matrix(gap_aggregate, data_dims, rhos, metric) for metric, _ in specs]
    abs_max = max(
        max(abs(value) for value in matrix[~torch.isnan(matrix)].flatten().tolist())
        for matrix in matrices
        if (~torch.isnan(matrix)).any()
    )
    abs_max = max(abs_max, 1e-8)

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False, constrained_layout=True)
    for axis, (metric, title), matrix in zip(axes.flatten(), specs, matrices):
        image = axis.imshow(matrix.numpy(), aspect="auto", cmap="coolwarm", vmin=-abs_max, vmax=abs_max)
        axis.set_title(title)
        axis.set_xlabel("dimension p")
        axis.set_ylabel("rho")
        axis.set_xticks(range(len(data_dims)))
        axis.set_xticklabels(data_dims)
        axis.set_yticks(range(len(rhos)))
        axis.set_yticklabels([f"{rho:.2f}" for rho in rhos])

        for rho_index, rho in enumerate(rhos):
            for dim_index, data_dim in enumerate(data_dims):
                value = matrix[rho_index, dim_index].item()
                if value == value:
                    axis.text(dim_index, rho_index, f"{value:.3f}", ha="center", va="center", fontsize=8)

    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_reference_covariance_heatmaps(eval_samples, generated_samples, output_path):
    panel_specs = [
        ("Real", torch.cov(eval_samples.T)),
        ("Product marginals", torch.cov(generated_samples["Product marginals"].T)),
        ("WGAN", torch.cov(generated_samples["WGAN"].T)),
        (r"Best $W_1$ diagonal", torch.cov(generated_samples["Best W1 diagonal Gaussian"].T)),
        (r"$W_2$ diagonal", torch.cov(generated_samples["Best diagonal Gaussian"].T)),
    ]
    max_abs = max(matrix.abs().max().item() for _, matrix in panel_specs)

    figure = plt.figure(figsize=(10.2, 6.2))
    grid = figure.add_gridspec(
        2,
        6,
        left=0.055,
        right=0.865,
        bottom=0.075,
        top=0.94,
        wspace=0.28,
        hspace=0.34,
    )
    axes = [
        figure.add_subplot(grid[0, 0:2]),
        figure.add_subplot(grid[0, 2:4]),
        figure.add_subplot(grid[0, 4:6]),
        figure.add_subplot(grid[1, 1:3]),
        figure.add_subplot(grid[1, 3:5]),
    ]

    image = None
    tick_positions = [0, eval_samples.shape[1] // 2 - 1, eval_samples.shape[1] - 1]
    tick_labels = [str(position + 1) for position in tick_positions]
    for axis, (title, covariance) in zip(axes, panel_specs):
        image = axis.imshow(
            covariance.numpy(),
            cmap="coolwarm",
            vmin=-max_abs,
            vmax=max_abs,
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=11, pad=7)
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(tick_labels, fontsize=8)
        axis.set_yticks(tick_positions)
        axis.set_yticklabels(tick_labels, fontsize=8)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_linewidth(0.75)
            spine.set_color("#555555")

    colorbar_axis = figure.add_axes([0.895, 0.18, 0.024, 0.64])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.ax.set_title("covariance", fontsize=9, pad=8)
    colorbar.ax.tick_params(labelsize=8)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_summary_text(aggregate_results, training_aggregate, gap_aggregate, output_path):
    lines = []
    data_dims = sorted({row["data_dim"] for row in aggregate_results})
    rhos = sorted({row["rho"] for row in aggregate_results})

    for data_dim in data_dims:
        lines.append(f"p = {data_dim}")
        for rho in rhos:
            lines.append(f"  rho = {rho:.2f}")
            for method in METHOD_ORDER:
                method_rows = [
                    row
                    for row in aggregate_results
                    if row["data_dim"] == data_dim and row["rho"] == rho and row["method"] == method
                ]
                if not method_rows:
                    continue

                row = method_rows[0]
                lines.extend(
                    [
                        f"    {method}",
                        f"      W1: {row['w1_mean']:.4f} +/- {row['w1_std']:.4f}",
                        f"      W2: {row['w2_mean']:.4f} +/- {row['w2_std']:.4f}",
                        f"      mean_error: {row['mean_error_mean']:.4f} +/- {row['mean_error_std']:.4f}",
                        f"      diag_var_error: {row['diag_var_error_mean']:.4f} +/- {row['diag_var_error_std']:.4f}",
                        f"      candidate_diag_var_mean: {row['candidate_diag_var_mean_mean']:.4f} +/- {row['candidate_diag_var_mean_std']:.4f}",
                        f"      theoretical_diag_var: {row['theoretical_diag_var_mean']:.4f}",
                        f"      candidate_minus_theoretical_diag_var: {row['candidate_minus_theoretical_diag_var_mean']:.4f} +/- {row['candidate_minus_theoretical_diag_var_std']:.4f}",
                        f"      offdiag_cov_error: {row['offdiag_cov_error_mean']:.4f} +/- {row['offdiag_cov_error_std']:.4f}",
                        f"      cov_fro_error: {row['cov_fro_error_mean']:.4f} +/- {row['cov_fro_error_std']:.4f}",
                        f"      avg_abs_corr: {row['avg_abs_corr_mean']:.4f} +/- {row['avg_abs_corr_std']:.4f}",
                    ]
                )

            training_rows = [
                row for row in training_aggregate if row["data_dim"] == data_dim and row["rho"] == rho
            ]
            if training_rows:
                training_row = training_rows[0]
                lines.extend(
                    [
                        "    WGAN checkpointing",
                        f"      selected_epoch: {training_row['selected_epoch_mean']:.1f} +/- {training_row['selected_epoch_std']:.1f}",
                        f"      selected_eval_w1: {training_row['selected_eval_w1_mean']:.4f} +/- {training_row['selected_eval_w1_std']:.4f}",
                        f"      selected_eval_mean_error: {training_row['selected_eval_mean_error_mean']:.4f} +/- {training_row['selected_eval_mean_error_std']:.4f}",
                        f"      selected_eval_diag_var_error: {training_row['selected_eval_diag_var_error_mean']:.4f} +/- {training_row['selected_eval_diag_var_error_std']:.4f}",
                        f"      selected_checkpoint_score: {training_row['selected_checkpoint_score_mean']:.4f} +/- {training_row['selected_checkpoint_score_std']:.4f}",
                    ]
                )

            gap_rows = [row for row in gap_aggregate if row["data_dim"] == data_dim and row["rho"] == rho]
            if gap_rows:
                gap_row = gap_rows[0]
                lines.extend(
                    [
                        "    Paired transport gaps",
                        f"      product - best diagonal (W1): {gap_row['product_minus_diag_w1_mean']:.4f} +/- {gap_row['product_minus_diag_w1_std']:.4f}",
                        f"      product - best diagonal (W2): {gap_row['product_minus_diag_w2_mean']:.4f} +/- {gap_row['product_minus_diag_w2_std']:.4f}",
                        f"      W1 diagonal - best diagonal (W1): {gap_row['w1diag_minus_diag_w1_mean']:.4f} +/- {gap_row['w1diag_minus_diag_w1_std']:.4f}",
                        f"      W1 diagonal - best diagonal (W2): {gap_row['w1diag_minus_diag_w2_mean']:.4f} +/- {gap_row['w1diag_minus_diag_w2_std']:.4f}",
                        f"      product - W1 diagonal (W1): {gap_row['product_minus_w1diag_w1_mean']:.4f} +/- {gap_row['product_minus_w1diag_w1_std']:.4f}",
                        f"      product - W1 diagonal (W2): {gap_row['product_minus_w1diag_w2_mean']:.4f} +/- {gap_row['product_minus_w1diag_w2_std']:.4f}",
                        f"      WGAN - best diagonal (W1): {gap_row['wgan_minus_diag_w1_mean']:.4f} +/- {gap_row['wgan_minus_diag_w1_std']:.4f}",
                        f"      WGAN - best diagonal (W2): {gap_row['wgan_minus_diag_w2_mean']:.4f} +/- {gap_row['wgan_minus_diag_w2_std']:.4f}",
                        f"      WGAN - W1 diagonal (W1): {gap_row['wgan_minus_w1diag_w1_mean']:.4f} +/- {gap_row['wgan_minus_w1diag_w1_std']:.4f}",
                        f"      WGAN - W1 diagonal (W2): {gap_row['wgan_minus_w1diag_w2_mean']:.4f} +/- {gap_row['wgan_minus_w1diag_w2_std']:.4f}",
                    ]
                )

            lines.append("")

    output_path.write_text("\n".join(lines).strip() + "\n")


def run_single_experiment(args, data_dim, rho, seed, device):
    torch.manual_seed(seed)
    train_samples = make_equicorrelated_gaussian_samples(args.train_samples, data_dim, rho, seed)
    val_samples = make_equicorrelated_gaussian_samples(args.val_samples, data_dim, rho, seed + 1)
    eval_samples = make_equicorrelated_gaussian_samples(args.eval_samples, data_dim, rho, seed + 2)
    theoretical_diag_var = theoretical_equicorrelated_diagonal_variance(data_dim, rho)
    dataloader = make_dataloader(train_samples, args.batch_size, seed=seed)

    generator = FactorizedGenerator(
        data_dim=data_dim,
        latent_dim=args.latent_dim,
        hidden_dims=build_hidden_dims(args.generator_hidden_dim, args.generator_depth),
        activation=args.generator_activation,
    )
    critic = Critic(
        data_dim=data_dim,
        hidden_dims=build_hidden_dims(args.critic_hidden_dim, args.critic_depth),
        feature_map=args.critic_feature_map,
        activation=args.critic_activation,
    )
    plot_w1 = seed == args.reference_seed and args.w1_eval_period > 0
    checkpoint_selection_map = {
        "last": "last",
        "best_val_w1": "best_eval_w1",
        "best_val_composite": "best_eval_composite",
    }
    trainer_checkpoint_selection = checkpoint_selection_map[args.checkpoint_selection]
    select_best_checkpoint = trainer_checkpoint_selection != "last"
    track_w1 = plot_w1 or select_best_checkpoint
    lr_g, lr_c = resolve_learning_rates(args.lr, args.generator_lr, args.critic_lr)
    trainer = WGANTrainer(
        generator=generator,
        critic=critic,
        device=device,
        lr_g=lr_g,
        lr_c=lr_c,
        n_critic=args.n_critic,
        gp_lambda=args.gp_lambda,
        optimizer_name=args.optimizer,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
    )

    metric_seed = seed + 10_000 * data_dim + int(round(1_000 * (rho + 1.0)))
    history = trainer.fit(
        dataloader,
        args.num_epochs,
        train_metric_samples=train_samples if plot_w1 else None,
        eval_metric_samples=val_samples if track_w1 else None,
        metric_period=args.w1_eval_period if track_w1 else 0,
        metric_max_samples=args.ot_eval_samples,
        metric_seed=metric_seed,
        checkpoint_selection=trainer_checkpoint_selection,
        checkpoint_var_weight=args.checkpoint_var_weight,
        checkpoint_mean_weight=args.checkpoint_mean_weight,
    )
    wgan_samples = trainer.sample(args.eval_samples, seed=seed + 3).cpu()
    product_samples = sample_product_of_marginals(train_samples, args.eval_samples, seed=seed + 4).cpu()
    diagonal_fit = fit_best_diagonal_gaussian_w2(train_samples, num_steps=args.diag_fit_steps, lr=args.diag_fit_lr)
    diagonal_samples = sample_diagonal_gaussian(
        diagonal_fit["mean"], diagonal_fit["diag_vars"], args.eval_samples, seed=seed + 5
    ).cpu()
    w1_diagonal_fit = fit_best_scalar_diagonal_gaussian_w1(
        train_samples=train_samples,
        validation_samples=val_samples,
        min_var=args.w1_diag_fit_min_var,
        max_var=args.w1_diag_fit_max_var,
        grid_size=args.w1_diag_fit_grid_size,
        refine_rounds=args.w1_diag_fit_refine_rounds,
        ot_eval_samples=args.w1_diag_fit_ot_samples,
        seed=metric_seed + 7,
    )
    w1_diagonal_samples = sample_diagonal_gaussian(
        w1_diagonal_fit["mean"], w1_diagonal_fit["diag_vars"], args.eval_samples, seed=seed + 6
    ).cpu()

    generated_samples = {
        "Best diagonal Gaussian": diagonal_samples,
        "Best W1 diagonal Gaussian": w1_diagonal_samples,
        "Product marginals": product_samples,
        "WGAN": wgan_samples,
    }

    results = []
    for method_name in METHOD_ORDER:
        metrics = compute_sample_summary(
            eval_samples,
            generated_samples[method_name],
            args.ot_eval_samples,
            metric_seed,
            theoretical_diag_var,
        )
        metrics["data_dim"] = data_dim
        metrics["rho"] = rho
        metrics["seed"] = seed
        metrics["method"] = method_name
        results.append(metrics)

    training_result = {
        "data_dim": data_dim,
        "rho": rho,
        "seed": seed,
        "selected_epoch": float(history["selected_epoch"]),
        "selected_eval_w1": float(history["selected_eval_w1"]),
        "selected_eval_mean_error": float(history["selected_eval_mean_error"]),
        "selected_eval_diag_var_error": float(history["selected_eval_diag_var_error"]),
        "selected_checkpoint_score": float(history["selected_checkpoint_score"]),
        "best_epoch": float(history["best_epoch"]) if history["best_epoch"] is not None else float(history["selected_epoch"]),
        "best_eval_w1": float(history["best_eval_w1"])
        if history["best_epoch"] is not None
        else float(history["selected_eval_w1"]),
        "best_eval_mean_error": float(history["best_eval_mean_error"])
        if history["best_epoch"] is not None
        else float(history["selected_eval_mean_error"]),
        "best_eval_diag_var_error": float(history["best_eval_diag_var_error"])
        if history["best_epoch"] is not None
        else float(history["selected_eval_diag_var_error"]),
        "best_checkpoint_score": float(history["best_checkpoint_score"])
        if history["best_epoch"] is not None
        else float(history["selected_checkpoint_score"]),
    }

    results.sort(key=lambda item: item["w1"])

    return {
        "results": results,
        "training_result": training_result,
        "history": history,
        "eval_samples": eval_samples,
        "generated_samples": generated_samples,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep dimension and equicorrelation for Gaussian data and compare independent approximations."
    )
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=2000)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--data-dims", type=int, nargs="+", default=[2, 5, 10, 20])
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--generator-hidden-dim", type=int, default=64)
    parser.add_argument("--generator-depth", type=int, default=3)
    parser.add_argument("--generator-activation", choices=["relu", "silu", "gelu"], default="silu")
    parser.add_argument("--critic-hidden-dim", type=int, default=64)
    parser.add_argument("--critic-depth", type=int, default=2)
    parser.add_argument("--critic-feature-map", choices=["raw", "quadratic"], default=DEFAULT_CRITIC_FEATURE_MAP)
    parser.add_argument("--critic-activation", choices=["relu", "silu", "gelu", "leaky_relu"], default="leaky_relu")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--generator-lr", type=float)
    parser.add_argument("--critic-lr", type=float)
    parser.add_argument("--optimizer", choices=["rmsprop", "adam"], default="adam")
    parser.add_argument("--adam-beta1", type=float, default=0.0)
    parser.add_argument("--adam-beta2", type=float, default=0.9)
    parser.add_argument("--n-critic", type=int, default=DEFAULT_N_CRITIC)
    parser.add_argument("--gp-lambda", type=float, default=DEFAULT_GP_LAMBDA)
    parser.add_argument("--diag-fit-steps", type=int, default=1000)
    parser.add_argument("--diag-fit-lr", type=float, default=0.05)
    parser.add_argument("--w1-diag-fit-min-var", type=float, default=0.05)
    parser.add_argument("--w1-diag-fit-max-var", type=float, default=1.25)
    parser.add_argument("--w1-diag-fit-grid-size", type=int, default=25)
    parser.add_argument("--w1-diag-fit-refine-rounds", type=int, default=2)
    parser.add_argument("--w1-diag-fit-ot-samples", type=int, default=128)
    parser.add_argument("--ot-eval-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--w1-eval-period", type=int, default=25)
    parser.add_argument(
        "--checkpoint-selection",
        choices=["last", "best_val_w1", "best_val_composite"],
        default="best_val_w1",
    )
    parser.add_argument("--checkpoint-var-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint-mean-weight", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gaussian_dimension_sweep"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    if args.num_seeds < 1:
        raise ValueError(f"Expected num_seeds to be at least 1, got {args.num_seeds}")
    if args.w1_diag_fit_ot_samples < 1:
        raise ValueError(f"Expected w1_diag_fit_ot_samples to be positive, got {args.w1_diag_fit_ot_samples}")

    data_dims = sorted(set(args.data_dims))
    rhos = sorted(set(args.rhos))
    seeds = args.seeds if args.seeds else list(range(args.seed, args.seed + args.num_seeds))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_results = []
    training_results = []
    reference_seed = seeds[0]
    args.reference_seed = reference_seed

    for data_dim in data_dims:
        for rho in rhos:
            for seed in seeds:
                print(f"Running p={data_dim}, rho={rho:.2f}, seed={seed}")
                experiment = run_single_experiment(args, data_dim, rho, seed, device)
                per_seed_results.extend(experiment["results"])
                training_results.append(experiment["training_result"])

                if seed == reference_seed:
                    tag = rho_tag(rho)
                    prefix = f"p_{data_dim}_rho_{tag}_reference"
                    save_loss_plot(experiment["history"], args.output_dir / f"{prefix}_wgan_losses.png")
                    save_reference_covariance_heatmaps(
                        experiment["eval_samples"],
                        experiment["generated_samples"],
                        args.output_dir / f"{prefix}_covariances.png",
                    )
                    save_results_csv(experiment["results"], args.output_dir / f"{prefix}_results.csv")
                    save_seed_summary(experiment["results"], args.output_dir / f"{prefix}_summary.txt")

                print(
                    " | ".join(
                        [
                            f"{row['method']}: W1={row['w1']:.4f}, W2={row['w2']:.4f}, avg_abs_corr={row['avg_abs_corr']:.4f}"
                            for row in experiment["results"]
                        ]
                    )
                )

    aggregate_results = aggregate_method_results(per_seed_results)
    training_aggregate = aggregate_training_results(training_results)
    gap_results = compute_gap_results(per_seed_results)
    gap_aggregate = aggregate_gap_results(gap_results)

    save_results_csv(per_seed_results, args.output_dir / "per_seed_results.csv")
    save_results_csv(aggregate_results, args.output_dir / "aggregate_results.csv")
    save_results_csv(training_results, args.output_dir / "wgan_training_per_seed.csv")
    save_results_csv(training_aggregate, args.output_dir / "wgan_training_aggregate.csv")
    save_results_csv(gap_results, args.output_dir / "paired_gap_per_seed.csv")
    save_results_csv(gap_aggregate, args.output_dir / "paired_gap_aggregate.csv")
    save_metric_grid(aggregate_results, args.output_dir / "metric_grid.png")
    save_variance_target_grid(aggregate_results, args.output_dir / "variance_targets.png")
    save_gap_heatmaps(gap_aggregate, args.output_dir / "gap_heatmaps.png")
    save_summary_text(aggregate_results, training_aggregate, gap_aggregate, args.output_dir / "summary.txt")

    print(f"Finished. Sweep outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
