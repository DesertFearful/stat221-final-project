import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from experiment_2d_gaussian import (
    build_hidden_dims,
    make_correlated_gaussian_samples,
    make_dataloader,
    resolve_device,
    resolve_learning_rates,
    save_loss_plot,
)
from src.baselines import fit_best_diagonal_gaussian_w2, sample_diagonal_gaussian, sample_product_of_marginals
from src.model import Critic, FactorizedGenerator
from src.ot_metrics import estimate_pot_wasserstein
from src.train_wgan import WGANTrainer


METHOD_ORDER = ["Best diagonal Gaussian", "Product marginals", "WGAN"]
METHOD_COLORS = {
    "Best diagonal Gaussian": "tab:green",
    "Product marginals": "tab:orange",
    "WGAN": "tab:blue",
}


# Keep the rho-dependent filenames readable and stable.
def rho_tag(rho):
    rho_string = f"{rho:.3f}".rstrip("0").rstrip(".")
    if not rho_string:
        rho_string = "0"
    return rho_string.replace("-", "m").replace(".", "p")


# In the symmetric unit-variance Gaussian case, the diagonal Gaussian optimum has a closed form.
def theoretical_diagonal_variance(rho):
    return 0.5 * (1.0 + math.sqrt(max(0.0, 1.0 - rho * rho)))


# Measure transport quality and low-order moments for one candidate method.
def compute_sample_summary(real_samples, candidate_samples, ot_eval_samples, seed):
    real_mean = real_samples.mean(dim=0)
    candidate_mean = candidate_samples.mean(dim=0)
    real_cov = torch.cov(real_samples.T)
    candidate_cov = torch.cov(candidate_samples.T)
    candidate_corr = torch.corrcoef(candidate_samples.T)[0, 1]
    ot_metrics = estimate_pot_wasserstein(real_samples, candidate_samples, max_samples=ot_eval_samples, seed=seed)

    return {
        "mean_error": torch.norm(candidate_mean - real_mean).item(),
        "var_error": 0.5 * (
            abs(candidate_cov[0, 0].item() - real_cov[0, 0].item()) +
            abs(candidate_cov[1, 1].item() - real_cov[1, 1].item())
        ),
        "cov_12": candidate_cov[0, 1].item(),
        "corr": candidate_corr.item(),
        "w1": ot_metrics["w1"],
        "w2": ot_metrics["w2"],
    }


# Save rows to CSV without adding a pandas dependency.
def save_results_csv(results, output_path):
    if not results:
        return

    fieldnames = list(results[0].keys())
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


# Save one scatter figure per rho for the first seed so you can inspect the geometry.
def save_reference_scatter(eval_samples, generated_samples, output_path):
    figure, axes = plt.subplots(1, len(METHOD_ORDER), figsize=(5 * len(METHOD_ORDER), 4), sharex=True, sharey=True)
    eval_array = eval_samples.numpy()
    x_min = eval_array[:, 0].min()
    x_max = eval_array[:, 0].max()
    y_min = eval_array[:, 1].min()
    y_max = eval_array[:, 1].max()

    for axis, method_name in zip(axes, METHOD_ORDER):
        candidate_array = generated_samples[method_name].numpy()
        axis.scatter(eval_array[:, 0], eval_array[:, 1], s=4, alpha=0.12, color="gray")
        axis.scatter(candidate_array[:, 0], candidate_array[:, 1], s=4, alpha=0.35, color=METHOD_COLORS[method_name])
        axis.set_title(method_name)
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x1")

    axes[0].set_ylabel("x2")
    figure.suptitle("Evaluation samples in gray, generated samples in color")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


# Average the method metrics across seeds separately for each rho.
def aggregate_method_results(per_seed_results):
    metrics = ["w1", "w2", "mean_error", "var_error", "corr", "cov_12"]
    grouped = {}

    for result in per_seed_results:
        key = (result["rho"], result["method"])
        grouped.setdefault(key, []).append(result)

    aggregate = []
    for rho in sorted({result["rho"] for result in per_seed_results}):
        for method in METHOD_ORDER:
            key = (rho, method)
            if key not in grouped:
                continue

            group = grouped[key]
            summary = {"rho": rho, "method": method, "num_seeds": len(group)}
            for metric in metrics:
                values = torch.tensor([item[metric] for item in group], dtype=torch.float64)
                summary[f"{metric}_mean"] = values.mean().item()
                summary[f"{metric}_std"] = values.std(unbiased=False).item()

            aggregate.append(summary)

    return aggregate


# Track how the fitted diagonal Gaussian changes with dependence strength.
def aggregate_diagonal_results(diagonal_results):
    metrics = ["diag_var_1", "diag_var_2", "diag_var_mean", "gaussian_w2"]
    grouped = {}

    for result in diagonal_results:
        grouped.setdefault(result["rho"], []).append(result)

    aggregate = []
    for rho in sorted(grouped):
        group = grouped[rho]
        summary = {
            "rho": rho,
            "num_seeds": len(group),
            "theoretical_diag_var": group[0]["theoretical_diag_var"],
        }

        for metric in metrics:
            values = torch.tensor([item[metric] for item in group], dtype=torch.float64)
            summary[f"{metric}_mean"] = values.mean().item()
            summary[f"{metric}_std"] = values.std(unbiased=False).item()

        aggregate.append(summary)

    return aggregate


# Compare methods on the same seed so the gap curves are paired and easier to trust.
def compute_gap_results(per_seed_results):
    grouped = {}

    for result in per_seed_results:
        key = (result["rho"], result["seed"])
        grouped.setdefault(key, {})[result["method"]] = result

    gap_results = []
    for (rho, seed), methods in sorted(grouped.items()):
        if not all(method in methods for method in METHOD_ORDER):
            continue

        diagonal = methods["Best diagonal Gaussian"]
        product = methods["Product marginals"]
        wgan = methods["WGAN"]
        gap_results.append(
            {
                "rho": rho,
                "seed": seed,
                "product_minus_diag_w1": product["w1"] - diagonal["w1"],
                "product_minus_diag_w2": product["w2"] - diagonal["w2"],
                "wgan_minus_diag_w1": wgan["w1"] - diagonal["w1"],
                "wgan_minus_diag_w2": wgan["w2"] - diagonal["w2"],
            }
        )

    return gap_results


# Average the paired method gaps for each rho.
def aggregate_gap_results(gap_results):
    metrics = [
        "product_minus_diag_w1",
        "product_minus_diag_w2",
        "wgan_minus_diag_w1",
        "wgan_minus_diag_w2",
    ]
    grouped = {}

    for result in gap_results:
        grouped.setdefault(result["rho"], []).append(result)

    aggregate = []
    for rho in sorted(grouped):
        group = grouped[rho]
        summary = {"rho": rho, "num_seeds": len(group)}
        for metric in metrics:
            values = torch.tensor([item[metric] for item in group], dtype=torch.float64)
            summary[f"{metric}_mean"] = values.mean().item()
            summary[f"{metric}_std"] = values.std(unbiased=False).item()

        aggregate.append(summary)

    return aggregate


# Summarize the main method metrics in one figure so you can see the dependence trend at a glance.
def save_summary_curves(aggregate_results, output_path):
    metrics = ["w1", "w2", "mean_error", "var_error"]
    titles = {
        "w1": "W1 vs rho",
        "w2": "W2 vs rho",
        "mean_error": "Mean error vs rho",
        "var_error": "Variance error vs rho",
    }

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    for axis, metric in zip(axes, metrics):
        for method in METHOD_ORDER:
            method_rows = [row for row in aggregate_results if row["method"] == method]
            if not method_rows:
                continue

            x_values = [row["rho"] for row in method_rows]
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

        axis.set_title(titles[metric])
        axis.set_xlabel("rho")
        axis.grid(alpha=0.25)

    axes[0].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


# Show the learned diagonal variances against the closed-form Gaussian benchmark.
def save_diagonal_variance_plot(diagonal_aggregate, output_path):
    rho_values = [row["rho"] for row in diagonal_aggregate]
    var1 = [row["diag_var_1_mean"] for row in diagonal_aggregate]
    var1_std = [row["diag_var_1_std"] for row in diagonal_aggregate]
    var2 = [row["diag_var_2_mean"] for row in diagonal_aggregate]
    var2_std = [row["diag_var_2_std"] for row in diagonal_aggregate]
    theory = [row["theoretical_diag_var"] for row in diagonal_aggregate]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(rho_values, var1, yerr=var1_std, marker="o", capsize=4, label="Fitted variance x1", color="tab:green")
    axis.errorbar(rho_values, var2, yerr=var2_std, marker="s", capsize=4, label="Fitted variance x2", color="tab:olive")
    axis.plot(rho_values, theory, linestyle="--", color="black", label="Theoretical optimum")
    axis.axhline(1.0, linestyle=":", color="gray", label="Product marginals variance")
    axis.set_title("Best diagonal Gaussian variance vs rho")
    axis.set_xlabel("rho")
    axis.set_ylabel("variance")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


# Plot the paired transport gap relative to the best diagonal Gaussian.
def save_gap_plot(gap_aggregate, output_path):
    rho_values = [row["rho"] for row in gap_aggregate]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    gap_specs = [
        ("product_minus_diag_w1", "wgan_minus_diag_w1", "Delta W1 vs rho"),
        ("product_minus_diag_w2", "wgan_minus_diag_w2", "Delta W2 vs rho"),
    ]

    for axis, (product_metric, wgan_metric, title) in zip(axes, gap_specs):
        product_means = [row[f"{product_metric}_mean"] for row in gap_aggregate]
        product_stds = [row[f"{product_metric}_std"] for row in gap_aggregate]
        wgan_means = [row[f"{wgan_metric}_mean"] for row in gap_aggregate]
        wgan_stds = [row[f"{wgan_metric}_std"] for row in gap_aggregate]

        axis.errorbar(
            rho_values,
            product_means,
            yerr=product_stds,
            marker="o",
            capsize=4,
            label="Product marginals - best diagonal",
            color=METHOD_COLORS["Product marginals"],
        )
        axis.errorbar(
            rho_values,
            wgan_means,
            yerr=wgan_stds,
            marker="s",
            capsize=4,
            label="WGAN - best diagonal",
            color=METHOD_COLORS["WGAN"],
        )
        axis.axhline(0.0, linestyle="--", color="black", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("rho")
        axis.grid(alpha=0.25)

    axes[0].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


# Write a readable text summary that matches the CSV outputs.
def save_summary_text(aggregate_results, diagonal_aggregate, gap_aggregate, output_path):
    lines = []
    rho_values = sorted({row["rho"] for row in aggregate_results})

    for rho in rho_values:
        lines.append(f"rho = {rho:.3f}")
        for method in METHOD_ORDER:
            method_rows = [row for row in aggregate_results if row["rho"] == rho and row["method"] == method]
            if not method_rows:
                continue

            row = method_rows[0]
            lines.extend(
                [
                    f"  {method}",
                    f"    W1: {row['w1_mean']:.4f} +/- {row['w1_std']:.4f}",
                    f"    W2: {row['w2_mean']:.4f} +/- {row['w2_std']:.4f}",
                    f"    mean_error: {row['mean_error_mean']:.4f} +/- {row['mean_error_std']:.4f}",
                    f"    var_error: {row['var_error_mean']:.4f} +/- {row['var_error_std']:.4f}",
                    f"    corr: {row['corr_mean']:.4f} +/- {row['corr_std']:.4f}",
                ]
            )

        diagonal_row = [row for row in diagonal_aggregate if row["rho"] == rho][0]
        gap_row = [row for row in gap_aggregate if row["rho"] == rho][0]
        lines.extend(
            [
                "  Diagonal Gaussian fit",
                f"    x1 variance: {diagonal_row['diag_var_1_mean']:.4f} +/- {diagonal_row['diag_var_1_std']:.4f}",
                f"    x2 variance: {diagonal_row['diag_var_2_mean']:.4f} +/- {diagonal_row['diag_var_2_std']:.4f}",
                f"    theoretical variance: {diagonal_row['theoretical_diag_var']:.4f}",
                f"    Gaussian W2: {diagonal_row['gaussian_w2_mean']:.4f} +/- {diagonal_row['gaussian_w2_std']:.4f}",
                "  Paired transport gaps",
                f"    product - best diagonal (W1): {gap_row['product_minus_diag_w1_mean']:.4f} +/- {gap_row['product_minus_diag_w1_std']:.4f}",
                f"    product - best diagonal (W2): {gap_row['product_minus_diag_w2_mean']:.4f} +/- {gap_row['product_minus_diag_w2_std']:.4f}",
                f"    WGAN - best diagonal (W1): {gap_row['wgan_minus_diag_w1_mean']:.4f} +/- {gap_row['wgan_minus_diag_w1_std']:.4f}",
                f"    WGAN - best diagonal (W2): {gap_row['wgan_minus_diag_w2_mean']:.4f} +/- {gap_row['wgan_minus_diag_w2_std']:.4f}",
                "",
            ]
        )

    output_path.write_text("\n".join(lines).strip() + "\n")


# Run one rho/seed experiment and return both the metric rows and the fitted diagonal details.
def run_single_experiment(args, rho, seed, device):
    torch.manual_seed(seed)
    train_samples = make_correlated_gaussian_samples(args.train_samples, rho, seed)
    val_samples = make_correlated_gaussian_samples(args.val_samples, rho, seed + 1)
    eval_samples = make_correlated_gaussian_samples(args.eval_samples, rho, seed + 2)
    dataloader = make_dataloader(train_samples, args.batch_size, seed=seed)

    generator = FactorizedGenerator(
        data_dim=2,
        latent_dim=args.latent_dim,
        hidden_dims=build_hidden_dims(args.generator_hidden_dim, args.generator_depth),
        activation=args.generator_activation,
    )
    critic = Critic(
        data_dim=2,
        hidden_dims=build_hidden_dims(args.critic_hidden_dim, args.critic_depth),
        feature_map=args.critic_feature_map,
        activation=args.critic_activation,
    )
    plot_w1 = seed == args.reference_seed and args.w1_eval_period > 0
    select_best_checkpoint = args.checkpoint_selection == "best_val_w1"
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

    history = trainer.fit(
        dataloader,
        args.num_epochs,
        train_metric_samples=train_samples if plot_w1 else None,
        eval_metric_samples=val_samples if track_w1 else None,
        metric_period=args.w1_eval_period if track_w1 else 0,
        metric_max_samples=args.ot_eval_samples,
        metric_seed=seed + int(round(1000 * (rho + 1.0))),
        checkpoint_selection="best_eval_w1" if select_best_checkpoint else "last",
    )
    wgan_samples = trainer.sample(args.eval_samples, seed=seed + 3).cpu()
    product_samples = sample_product_of_marginals(train_samples, args.eval_samples, seed=seed + 4).cpu()
    diagonal_fit = fit_best_diagonal_gaussian_w2(train_samples, num_steps=args.diag_fit_steps, lr=args.diag_fit_lr)
    diagonal_samples = sample_diagonal_gaussian(
        diagonal_fit["mean"], diagonal_fit["diag_vars"], args.eval_samples, seed=seed + 5
    ).cpu()

    generated_samples = {
        "Best diagonal Gaussian": diagonal_samples,
        "Product marginals": product_samples,
        "WGAN": wgan_samples,
    }

    results = []
    ot_seed = seed + int(round(1000 * (rho + 1.0)))
    for method_name in METHOD_ORDER:
        metrics = compute_sample_summary(eval_samples, generated_samples[method_name], args.ot_eval_samples, ot_seed)
        metrics["rho"] = rho
        metrics["seed"] = seed
        metrics["method"] = method_name
        results.append(metrics)

    diag_vars = diagonal_fit["diag_vars"].cpu()
    diagonal_result = {
        "rho": rho,
        "seed": seed,
        "diag_var_1": diag_vars[0].item(),
        "diag_var_2": diag_vars[1].item(),
        "diag_var_mean": diag_vars.mean().item(),
        "gaussian_w2": math.sqrt(max(diagonal_fit["w2_squared"], 0.0)),
        "theoretical_diag_var": theoretical_diagonal_variance(rho),
    }

    return {
        "results": results,
        "diagonal_result": diagonal_result,
        "history": history,
        "eval_samples": eval_samples,
        "generated_samples": generated_samples,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep rho and compare WGAN, product marginals, and the best diagonal Gaussian.")
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=2000)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--generator-hidden-dim", type=int, default=64)
    parser.add_argument("--generator-depth", type=int, default=3)
    parser.add_argument("--generator-activation", choices=["relu", "silu", "gelu"], default="silu")
    parser.add_argument("--critic-hidden-dim", type=int, default=64)
    parser.add_argument("--critic-depth", type=int, default=2)
    parser.add_argument("--critic-feature-map", choices=["raw", "quadratic"], default="raw")
    parser.add_argument("--critic-activation", choices=["relu", "silu", "gelu", "leaky_relu"], default="leaky_relu")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--generator-lr", type=float)
    parser.add_argument("--critic-lr", type=float)
    parser.add_argument("--optimizer", choices=["rmsprop", "adam"], default="adam")
    parser.add_argument("--adam-beta1", type=float, default=0.0)
    parser.add_argument("--adam-beta2", type=float, default=0.9)
    parser.add_argument("--n-critic", type=int, default=5)
    parser.add_argument("--gp-lambda", type=float, default=10.0)
    parser.add_argument("--diag-fit-steps", type=int, default=1000)
    parser.add_argument("--diag-fit-lr", type=float, default=0.05)
    parser.add_argument("--ot-eval-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--w1-eval-period", type=int, default=10)
    parser.add_argument("--checkpoint-selection", choices=["last", "best_val_w1"], default="best_val_w1")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rho_comparison_sweep"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    if args.num_seeds < 1:
        raise ValueError(f"Expected num_seeds to be at least 1, got {args.num_seeds}")

    rhos = sorted(set(args.rhos))
    seeds = args.seeds if args.seeds else list(range(args.seed, args.seed + args.num_seeds))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_results = []
    diagonal_results = []
    reference_seed = seeds[0]
    args.reference_seed = reference_seed

    for rho in rhos:
        for seed in seeds:
            print(f"Running rho={rho:.3f}, seed={seed}")
            experiment = run_single_experiment(args, rho, seed, device)
            per_seed_results.extend(experiment["results"])
            diagonal_results.append(experiment["diagonal_result"])

            if seed == reference_seed:
                tag = rho_tag(rho)
                save_loss_plot(experiment["history"], args.output_dir / f"rho_{tag}_reference_wgan_losses.png")
                save_reference_scatter(
                    experiment["eval_samples"],
                    experiment["generated_samples"],
                    args.output_dir / f"rho_{tag}_reference_scatter.png",
                )

            print(
                " | ".join(
                    [
                        f"{row['method']}: W1={row['w1']:.4f}, W2={row['w2']:.4f}, corr={row['corr']:.4f}"
                        for row in experiment["results"]
                    ]
                )
            )

    aggregate_results = aggregate_method_results(per_seed_results)
    gap_results = compute_gap_results(per_seed_results)
    gap_aggregate = aggregate_gap_results(gap_results)
    diagonal_aggregate = aggregate_diagonal_results(diagonal_results)

    save_results_csv(per_seed_results, args.output_dir / "per_seed_results.csv")
    save_results_csv(aggregate_results, args.output_dir / "aggregate_results.csv")
    save_results_csv(diagonal_results, args.output_dir / "diagonal_fit_per_seed.csv")
    save_results_csv(diagonal_aggregate, args.output_dir / "diagonal_fit_aggregate.csv")
    save_results_csv(gap_results, args.output_dir / "paired_gap_per_seed.csv")
    save_results_csv(gap_aggregate, args.output_dir / "paired_gap_aggregate.csv")
    save_summary_curves(aggregate_results, args.output_dir / "summary_curves.png")
    save_diagonal_variance_plot(diagonal_aggregate, args.output_dir / "diag_variance_vs_rho.png")
    save_gap_plot(gap_aggregate, args.output_dir / "gap_vs_rho.png")
    save_summary_text(aggregate_results, diagonal_aggregate, gap_aggregate, args.output_dir / "summary.txt")

    print(f"Finished. Sweep outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
