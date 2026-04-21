import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from experiment_2d_gaussian import (
    make_correlated_gaussian_samples,
    make_dataloader,
    resolve_device,
    save_loss_plot,
    save_marginal_plot,
    save_scatter_plot,
    summarize_samples,
)
from src.model import Critic, FactorizedGenerator
from src.ot_metrics import estimate_pot_wasserstein
from src.train_wgan import WGANTrainer


# Format learning rates into stable, readable tags for filenames.
def lr_tag(lr):
    lr_string = f"{lr:.6g}"
    return lr_string.replace("-", "m").replace(".", "p")


# Compute the held-out sample metrics for one trained WGAN.
def compute_metrics(real_samples, fake_samples, history, ot_eval_samples, ot_seed):
    real_mean = real_samples.mean(dim=0)
    fake_mean = fake_samples.mean(dim=0)
    real_cov = torch.cov(real_samples.T)
    fake_cov = torch.cov(fake_samples.T)
    real_corr = torch.corrcoef(real_samples.T)[0, 1]
    fake_corr = torch.corrcoef(fake_samples.T)[0, 1]
    ot_metrics = estimate_pot_wasserstein(real_samples, fake_samples, max_samples=ot_eval_samples, seed=ot_seed)

    return {
        "real_mean_1": real_mean[0].item(),
        "real_mean_2": real_mean[1].item(),
        "fake_mean_1": fake_mean[0].item(),
        "fake_mean_2": fake_mean[1].item(),
        "real_var_1": real_cov[0, 0].item(),
        "real_var_2": real_cov[1, 1].item(),
        "fake_var_1": fake_cov[0, 0].item(),
        "fake_var_2": fake_cov[1, 1].item(),
        "real_cov_12": real_cov[0, 1].item(),
        "fake_cov_12": fake_cov[0, 1].item(),
        "real_corr": real_corr.item(),
        "fake_corr": fake_corr.item(),
        "final_generator_loss": history["generator_loss"][-1],
        "final_critic_loss": history["critic_loss"][-1],
        "selected_epoch": history["selected_epoch"],
        "best_epoch": history["best_epoch"],
        "best_val_w1": history["best_eval_w1"],
        "selected_val_w1": history["selected_eval_w1"],
        "test_w1": ot_metrics["w1"],
        "test_w2": ot_metrics["w2"],
        "mean_error": torch.norm(fake_mean - real_mean).item(),
        "var_error": 0.5 * (
            abs(fake_cov[0, 0].item() - real_cov[0, 0].item()) +
            abs(fake_cov[1, 1].item() - real_cov[1, 1].item())
        ),
    }


# Save rows to CSV without bringing in extra dependencies.
def save_results_csv(results, output_path):
    if not results:
        return

    fieldnames = list(results[0].keys())
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


# Aggregate per-seed results into mean and std for each optimizer/learning-rate pair.
def aggregate_results(per_seed_results):
    metrics = [
        "best_val_w1",
        "selected_val_w1",
        "test_w1",
        "test_w2",
        "mean_error",
        "var_error",
        "fake_corr",
        "selected_epoch",
    ]
    aggregate = []

    optimizers = sorted({result["optimizer"] for result in per_seed_results})

    for optimizer in optimizers:
        optimizer_results = [result for result in per_seed_results if result["optimizer"] == optimizer]
        for lr in sorted({result["lr"] for result in optimizer_results}):
            group = [result for result in optimizer_results if result["lr"] == lr]
            summary = {"optimizer": optimizer, "lr": lr, "num_seeds": len(group)}

            for metric in metrics:
                values = torch.tensor([result[metric] for result in group], dtype=torch.float64)
                summary[f"{metric}_mean"] = values.mean().item()
                summary[f"{metric}_std"] = values.std(unbiased=False).item()

            aggregate.append(summary)

    return aggregate


# Plot the main sweep diagnostics on a log-LR axis, with one curve per optimizer.
def save_summary_curves(aggregate_results, output_path):
    metrics = [
        ("best_val_w1", "Best validation W1"),
        ("test_w1", "Test W1"),
        ("test_w2", "Test W2"),
        ("var_error", "Variance error"),
        ("mean_error", "Mean error"),
        ("selected_epoch", "Selected epoch"),
    ]

    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    optimizers = sorted({row["optimizer"] for row in aggregate_results})

    for axis, (metric, title) in zip(axes, metrics):
        for optimizer in optimizers:
            rows = [row for row in aggregate_results if row["optimizer"] == optimizer]
            rows.sort(key=lambda row: row["lr"])
            lrs = [row["lr"] for row in rows]
            means = [row[f"{metric}_mean"] for row in rows]
            stds = [row[f"{metric}_std"] for row in rows]
            axis.errorbar(lrs, means, yerr=stds, marker="o", capsize=4, label=optimizer)

        axis.set_xscale("log")
        axis.set_title(title)
        axis.set_xlabel("learning rate")
        axis.grid(alpha=0.25)

    axes[0].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


# Write a compact text summary for quick inspection in the terminal or editor.
def save_text_summary(aggregate_results, output_path):
    lines = []

    for optimizer in sorted({row["optimizer"] for row in aggregate_results}):
        lines.append(f"optimizer = {optimizer}")
        lines.append("")

        rows = [row for row in aggregate_results if row["optimizer"] == optimizer]
        rows.sort(key=lambda row: row["lr"])

        for row in rows:
            lines.extend(
                [
                    f"lr = {row['lr']:.6g}",
                    f"  best_val_w1: {row['best_val_w1_mean']:.4f} +/- {row['best_val_w1_std']:.4f}",
                    f"  test_w1: {row['test_w1_mean']:.4f} +/- {row['test_w1_std']:.4f}",
                    f"  test_w2: {row['test_w2_mean']:.4f} +/- {row['test_w2_std']:.4f}",
                    f"  mean_error: {row['mean_error_mean']:.4f} +/- {row['mean_error_std']:.4f}",
                    f"  var_error: {row['var_error_mean']:.4f} +/- {row['var_error_std']:.4f}",
                    f"  fake_corr: {row['fake_corr_mean']:.4f} +/- {row['fake_corr_std']:.4f}",
                    f"  selected_epoch: {row['selected_epoch_mean']:.1f} +/- {row['selected_epoch_std']:.1f}",
                    f"  num_seeds: {row['num_seeds']}",
                    "",
                ]
            )

    output_path.write_text("\n".join(lines).strip() + "\n")


# Train one WGAN for one seed, one optimizer, and one learning rate.
def run_single_seed(args, optimizer_name, lr, seed, device):
    torch.manual_seed(seed)
    train_samples = make_correlated_gaussian_samples(args.train_samples, args.rho, seed)
    val_samples = make_correlated_gaussian_samples(args.val_samples, args.rho, seed + 1)
    eval_samples = make_correlated_gaussian_samples(args.eval_samples, args.rho, seed + 2)
    dataloader = make_dataloader(train_samples, args.batch_size, seed=seed)

    generator = FactorizedGenerator(data_dim=2, latent_dim=args.latent_dim, hidden_dim=args.generator_hidden_dim)
    critic = Critic(data_dim=2, hidden_dim=args.critic_hidden_dim, feature_map=args.critic_feature_map)
    trainer = WGANTrainer(
        generator=generator,
        critic=critic,
        device=device,
        lr=lr,
        n_critic=args.n_critic,
        weight_clip=args.weight_clip,
        optimizer_name=optimizer_name,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
    )

    history = trainer.fit(
        dataloader,
        args.num_epochs,
        train_metric_samples=train_samples,
        eval_metric_samples=val_samples,
        metric_period=args.w1_eval_period,
        metric_max_samples=args.ot_eval_samples,
        metric_seed=seed,
        checkpoint_selection="best_eval_w1",
    )
    fake_samples = trainer.sample(args.eval_samples, seed=seed + 3).cpu()

    metrics = compute_metrics(eval_samples, fake_samples, history, args.ot_eval_samples, seed)
    metrics["optimizer"] = optimizer_name
    metrics["lr"] = lr
    metrics["seed"] = seed
    metrics["adam_beta1"] = args.adam_beta1 if optimizer_name == "adam" else float("nan")
    metrics["adam_beta2"] = args.adam_beta2 if optimizer_name == "adam" else float("nan")

    return {
        "history": history,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "fake_samples": fake_samples,
        "metrics": metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep the WGAN optimizer and learning rate on the 2D correlated Gaussian task.")
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
    parser.add_argument("--optimizer", choices=["rmsprop", "adam"])
    parser.add_argument("--optimizers", choices=["rmsprop", "adam"], nargs="+")
    parser.add_argument("--adam-beta1", type=float, default=0.0)
    parser.add_argument("--adam-beta2", type=float, default=0.9)
    parser.add_argument("--n-critic", type=int, default=5)
    parser.add_argument("--weight-clip", type=float, default=0.05)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-5, 3e-5, 5e-5, 1e-4, 2e-4])
    parser.add_argument("--w1-eval-period", type=int, default=10)
    parser.add_argument("--ot-eval-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/learning_rate_sweep"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    if args.num_seeds < 1:
        raise ValueError(f"Expected num_seeds to be at least 1, got {args.num_seeds}")
    if args.w1_eval_period < 1:
        raise ValueError(f"Expected w1_eval_period to be at least 1, got {args.w1_eval_period}")
    if args.optimizer is not None and args.optimizers is not None:
        raise ValueError("Use either --optimizer or --optimizers, not both.")

    lrs = sorted(set(args.learning_rates))
    if args.optimizers is not None:
        optimizers = list(dict.fromkeys(args.optimizers))
    elif args.optimizer is not None:
        optimizers = [args.optimizer]
    else:
        optimizers = ["rmsprop"]
    seeds = args.seeds if args.seeds else list(range(args.seed, args.seed + args.num_seeds))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_results = []
    reference_seed = seeds[0]
    total_runs = len(optimizers) * len(lrs) * len(seeds)
    run_index = 0

    for optimizer_name in optimizers:
        optimizer_dir = args.output_dir / optimizer_name
        optimizer_dir.mkdir(parents=True, exist_ok=True)

        for lr in lrs:
            lr_dir = optimizer_dir / f"lr_{lr_tag(lr)}"
            lr_dir.mkdir(parents=True, exist_ok=True)

            for seed in seeds:
                run_index += 1
                print(f"[{run_index}/{total_runs}] Running optimizer={optimizer_name}, lr={lr:.6g}, seed={seed}")
                run = run_single_seed(args, optimizer_name, lr, seed, device)
                metrics = run["metrics"]
                per_seed_results.append(metrics)

                if seed == reference_seed:
                    save_loss_plot(run["history"], lr_dir / "reference_wgan_losses.png")
                    save_scatter_plot(run["eval_samples"], run["fake_samples"], lr_dir / "reference_scatter.png")
                    save_marginal_plot(run["eval_samples"], run["fake_samples"], lr_dir / "reference_marginals.png")
                    summarize_samples(
                        run["eval_samples"],
                        run["fake_samples"],
                        lr_dir / "reference_summary.txt",
                        extra_metrics={
                            "Best validation W1": metrics["best_val_w1"],
                            "Test W1": metrics["test_w1"],
                            "Test W2": metrics["test_w2"],
                        },
                    )
                    torch.save(run["history"], lr_dir / "reference_history.pt")
                    torch.save(run["fake_samples"], lr_dir / "reference_generated_samples.pt")

                print(
                    f"[{run_index}/{total_runs}] Finished optimizer={optimizer_name}, lr={lr:.6g}, seed={seed} | "
                    f"best_val_w1={metrics['best_val_w1']:.4f} | "
                    f"test_w1={metrics['test_w1']:.4f} | "
                    f"test_w2={metrics['test_w2']:.4f} | "
                    f"selected_epoch={metrics['selected_epoch']}"
                )

    aggregate = aggregate_results(per_seed_results)
    save_results_csv(per_seed_results, args.output_dir / "per_seed_results.csv")
    save_results_csv(aggregate, args.output_dir / "aggregate_results.csv")
    save_summary_curves(aggregate, args.output_dir / "summary_curves.png")
    save_text_summary(aggregate, args.output_dir / "summary.txt")

    print("Aggregate results:")
    for row in aggregate:
        print(
            f"optimizer={row['optimizer']}, lr={row['lr']:.6g}: "
            f"best_val_w1={row['best_val_w1_mean']:.4f} +/- {row['best_val_w1_std']:.4f}, "
            f"test_w1={row['test_w1_mean']:.4f} +/- {row['test_w1_std']:.4f}, "
            f"test_w2={row['test_w2_mean']:.4f} +/- {row['test_w2_std']:.4f}, "
            f"var_error={row['var_error_mean']:.4f} +/- {row['var_error_std']:.4f}, "
            f"selected_epoch={row['selected_epoch_mean']:.1f} +/- {row['selected_epoch_std']:.1f}"
        )

    print(f"Finished. Sweep outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
