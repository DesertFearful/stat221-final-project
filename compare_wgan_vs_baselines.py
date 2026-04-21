import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from experiment_2d_gaussian import make_correlated_gaussian_samples, make_dataloader, resolve_device, save_loss_plot
from src.baselines import fit_best_diagonal_gaussian_w2, sample_diagonal_gaussian, sample_product_of_marginals
from src.model import Critic, FactorizedGenerator
from src.ot_metrics import estimate_pot_wasserstein
from src.train_wgan import WGANTrainer


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


def save_results_csv(results, output_path):
    if not results:
        return

    fieldnames = list(results[0].keys())
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def save_comparison_scatter(eval_samples, generated_samples, output_path):
    method_names = list(generated_samples.keys())
    figure, axes = plt.subplots(1, len(method_names), figsize=(5 * len(method_names), 4), sharex=True, sharey=True)

    if len(method_names) == 1:
        axes = [axes]

    eval_array = eval_samples.numpy()
    x_min = eval_array[:, 0].min()
    x_max = eval_array[:, 0].max()
    y_min = eval_array[:, 1].min()
    y_max = eval_array[:, 1].max()

    for axis, method_name in zip(axes, method_names):
        candidate_array = generated_samples[method_name].numpy()
        axis.scatter(eval_array[:, 0], eval_array[:, 1], s=4, alpha=0.12, color="gray")
        axis.scatter(candidate_array[:, 0], candidate_array[:, 1], s=4, alpha=0.35, color="tab:blue")
        axis.set_title(method_name)
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x1")

    axes[0].set_ylabel("x2")
    figure.suptitle("Evaluation samples in gray, candidate samples in blue")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_metric_bars(results, output_path, title_prefix=""):
    metrics = ["w1", "w2", "mean_error", "var_error", "corr"]
    labels = [result["method"] for result in results]
    figure, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))

    for axis, metric in zip(axes, metrics):
        values = [result[metric] for result in results]
        axis.bar(labels, values)
        axis.set_title(f"{title_prefix}{metric}")
        axis.tick_params(axis="x", rotation=25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_aggregate_metric_bars(results, output_path):
    metrics = ["w1", "w2", "mean_error", "var_error", "corr"]
    labels = [result["method"] for result in results]
    figure, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))

    for axis, metric in zip(axes, metrics):
        means = [result[f"{metric}_mean"] for result in results]
        stds = [result[f"{metric}_std"] for result in results]
        axis.bar(labels, means, yerr=stds, capsize=4)
        axis.set_title(f"{metric} mean +/- std")
        axis.tick_params(axis="x", rotation=25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_text_summary(results, output_path, aggregate=False):
    lines = []
    for result in results:
        lines.append(result["method"])
        if aggregate:
            lines.extend(
                [
                    f"  W1: {result['w1_mean']:.4f} +/- {result['w1_std']:.4f}",
                    f"  W2: {result['w2_mean']:.4f} +/- {result['w2_std']:.4f}",
                    f"  mean_error: {result['mean_error_mean']:.4f} +/- {result['mean_error_std']:.4f}",
                    f"  var_error: {result['var_error_mean']:.4f} +/- {result['var_error_std']:.4f}",
                    f"  corr: {result['corr_mean']:.4f} +/- {result['corr_std']:.4f}",
                    f"  num_seeds: {result['num_seeds']}",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"  W1: {result['w1']:.4f}",
                    f"  W2: {result['w2']:.4f}",
                    f"  mean_error: {result['mean_error']:.4f}",
                    f"  var_error: {result['var_error']:.4f}",
                    f"  corr: {result['corr']:.4f}",
                    "",
                ]
            )

    output_path.write_text("\n".join(lines).strip() + "\n")


def aggregate_results(per_seed_results):
    metrics = ["w1", "w2", "mean_error", "var_error", "corr"]
    methods = sorted({result["method"] for result in per_seed_results})
    aggregate = []

    for method in methods:
        method_results = [result for result in per_seed_results if result["method"] == method]
        summary = {"method": method, "num_seeds": len(method_results)}

        for metric in metrics:
            values = torch.tensor([result[metric] for result in method_results], dtype=torch.float64)
            summary[f"{metric}_mean"] = values.mean().item()
            summary[f"{metric}_std"] = values.std(unbiased=False).item()

        aggregate.append(summary)

    aggregate.sort(key=lambda item: item["w1_mean"])
    return aggregate


def run_single_seed(args, seed, device, output_dir, artifact_prefix=None):
    torch.manual_seed(seed)
    train_samples = make_correlated_gaussian_samples(args.train_samples, args.rho, seed)
    val_samples = make_correlated_gaussian_samples(args.val_samples, args.rho, seed + 1)
    eval_samples = make_correlated_gaussian_samples(args.eval_samples, args.rho, seed + 2)
    dataloader = make_dataloader(train_samples, args.batch_size, seed=seed)

    generator = FactorizedGenerator(data_dim=2, latent_dim=args.latent_dim, hidden_dim=args.generator_hidden_dim)
    critic = Critic(data_dim=2, hidden_dim=args.critic_hidden_dim, feature_map=args.critic_feature_map)
    plot_w1 = artifact_prefix is not None and args.w1_eval_period > 0
    select_best_checkpoint = args.checkpoint_selection == "best_val_w1"
    track_w1 = select_best_checkpoint or plot_w1
    trainer = WGANTrainer(
        generator=generator,
        critic=critic,
        device=device,
        lr=args.lr,
        n_critic=args.n_critic,
        weight_clip=args.weight_clip,
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
        metric_seed=seed,
        checkpoint_selection="best_eval_w1" if select_best_checkpoint else "last",
    )
    wgan_samples = trainer.sample(args.eval_samples).cpu()
    product_samples = sample_product_of_marginals(train_samples, args.eval_samples, seed=seed + 3).cpu()
    diagonal_fit = fit_best_diagonal_gaussian_w2(train_samples, num_steps=args.diag_fit_steps, lr=args.diag_fit_lr)
    diagonal_samples = sample_diagonal_gaussian(
        diagonal_fit["mean"], diagonal_fit["diag_vars"], args.eval_samples, seed=seed + 4
    ).cpu()

    generated_samples = {
        "WGAN": wgan_samples,
        "Product marginals": product_samples,
        "Best diagonal Gaussian": diagonal_samples,
    }

    results = []
    for method_name, candidate_samples in generated_samples.items():
        metrics = compute_sample_summary(eval_samples, candidate_samples, args.ot_eval_samples, seed)
        metrics["method"] = method_name
        metrics["seed"] = seed
        results.append(metrics)

    results.sort(key=lambda item: item["w1"])

    if artifact_prefix is not None:
        prefix = f"{artifact_prefix}_" if artifact_prefix else ""
        save_loss_plot(history, output_dir / f"{prefix}wgan_losses.png")
        save_comparison_scatter(eval_samples, generated_samples, output_dir / f"{prefix}comparison_scatter.png")
        save_metric_bars(results, output_dir / f"{prefix}comparison_metrics.png")
        save_results_csv(results, output_dir / f"{prefix}comparison_results.csv")
        save_text_summary(results, output_dir / f"{prefix}comparison_summary.txt")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Compare WGAN against independent baselines on the 2D Gaussian task.")
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=2000)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--rho", type=float, default=0.8)
    parser.add_argument("--latent-dim", type=int, default=1)
    parser.add_argument("--generator-hidden-dim", type=int, default=64)
    parser.add_argument("--critic-hidden-dim", type=int, default=64)
    parser.add_argument("--critic-feature-map", choices=["raw", "quadratic"], default="raw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["rmsprop", "adam"], default="rmsprop")
    parser.add_argument("--adam-beta1", type=float, default=0.0)
    parser.add_argument("--adam-beta2", type=float, default=0.9)
    parser.add_argument("--n-critic", type=int, default=5)
    parser.add_argument("--weight-clip", type=float, default=0.05)
    parser.add_argument("--diag-fit-steps", type=int, default=1000)
    parser.add_argument("--diag-fit-lr", type=float, default=0.05)
    parser.add_argument("--ot-eval-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--w1-eval-period", type=int, default=10)
    parser.add_argument("--checkpoint-selection", choices=["last", "best_val_w1"], default="best_val_w1")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/wgan_baseline_comparison"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    if args.num_seeds < 1:
        raise ValueError(f"Expected num_seeds to be at least 1, got {args.num_seeds}")

    seeds = args.seeds if args.seeds else list(range(args.seed, args.seed + args.num_seeds))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    reference_seed = seeds[0]

    for seed in seeds:
        print(f"Running seed {seed}")
        artifact_prefix = None
        if len(seeds) == 1:
            artifact_prefix = ""
        elif seed == reference_seed:
            artifact_prefix = f"reference_seed_{seed}"

        seed_results = run_single_seed(args, seed, device, args.output_dir, artifact_prefix=artifact_prefix)
        all_results.extend(seed_results)

        print(f"Seed {seed} results:")
        for result in seed_results:
            print(
                f"{result['method']}: "
                f"W1={result['w1']:.4f}, "
                f"W2={result['w2']:.4f}, "
                f"mean_error={result['mean_error']:.4f}, "
                f"var_error={result['var_error']:.4f}, "
                f"corr={result['corr']:.4f}"
            )

    aggregate = aggregate_results(all_results)

    save_results_csv(all_results, args.output_dir / "per_seed_results.csv")
    save_results_csv(aggregate, args.output_dir / "aggregate_results.csv")
    save_aggregate_metric_bars(aggregate, args.output_dir / "aggregate_metrics.png")
    save_text_summary(aggregate, args.output_dir / "aggregate_summary.txt", aggregate=True)

    print("Aggregate results:")
    for result in aggregate:
        print(
            f"{result['method']}: "
            f"W1={result['w1_mean']:.4f} +/- {result['w1_std']:.4f}, "
            f"W2={result['w2_mean']:.4f} +/- {result['w2_std']:.4f}, "
            f"mean_error={result['mean_error_mean']:.4f} +/- {result['mean_error_std']:.4f}, "
            f"var_error={result['var_error_mean']:.4f} +/- {result['var_error_std']:.4f}, "
            f"corr={result['corr_mean']:.4f} +/- {result['corr_std']:.4f}"
        )

    print(f"Finished. Comparison outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
