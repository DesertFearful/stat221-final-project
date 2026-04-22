import argparse
import csv
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
    save_marginal_plot,
    save_scatter_plot,
    summarize_samples,
)
from src.model import Critic, FactorizedGenerator
from src.ot_metrics import estimate_pot_wasserstein
from src.train_wgan import WGANTrainer


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
        "best_val_w1": history["best_eval_w1"],
        "selected_val_w1": history["selected_eval_w1"],
        "w1": ot_metrics["w1"],
        "w2": ot_metrics["w2"],
        "mean_error": torch.norm(fake_mean - real_mean).item(),
        "var_error": 0.5 * (
            abs(fake_cov[0, 0].item() - real_cov[0, 0].item()) +
            abs(fake_cov[1, 1].item() - real_cov[1, 1].item())
        ),
    }


def save_results_csv(results, output_path):
    if not results:
        return

    fieldnames = list(results[0].keys())
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def aggregate_results(per_seed_results):
    metrics = [
        "best_val_w1",
        "selected_val_w1",
        "w1",
        "w2",
        "mean_error",
        "var_error",
        "fake_corr",
        "selected_epoch",
    ]
    grouped = {}

    for result in per_seed_results:
        key = (result["gp_lambda"], result["n_critic"])
        grouped.setdefault(key, []).append(result)

    aggregate = []
    for gp_lambda, n_critic in sorted(grouped):
        group = grouped[(gp_lambda, n_critic)]
        summary = {
            "gp_lambda": gp_lambda,
            "n_critic": n_critic,
            "num_seeds": len(group),
            "run_dir": group[0]["run_dir"],
        }

        for metric in metrics:
            values = torch.tensor([result[metric] for result in group], dtype=torch.float64)
            summary[f"{metric}_mean"] = values.mean().item()
            summary[f"{metric}_std"] = values.std(unbiased=False).item()

        aggregate.append(summary)

    return aggregate


def save_reference_grid(reference_eval_samples, aggregate_results, output_path):
    gp_lambdas = sorted({row["gp_lambda"] for row in aggregate_results})
    n_critics = sorted({row["n_critic"] for row in aggregate_results})
    results_map = {(row["gp_lambda"], row["n_critic"]): row for row in aggregate_results}

    real_array = reference_eval_samples.numpy()
    x_min = real_array[:, 0].min()
    x_max = real_array[:, 0].max()
    y_min = real_array[:, 1].min()
    y_max = real_array[:, 1].max()

    figure, axes = plt.subplots(
        len(gp_lambdas),
        len(n_critics),
        figsize=(4 * len(n_critics), 4 * len(gp_lambdas)),
        sharex=True,
        sharey=True,
    )

    if len(gp_lambdas) == 1 and len(n_critics) == 1:
        axes = [[axes]]
    elif len(gp_lambdas) == 1:
        axes = [axes]
    elif len(n_critics) == 1:
        axes = [[axis] for axis in axes]

    for row_index, gp_lambda in enumerate(gp_lambdas):
        for col_index, n_critic in enumerate(n_critics):
            axis = axes[row_index][col_index]
            result = results_map[(gp_lambda, n_critic)]
            fake_samples = torch.load(Path(result["run_dir"]) / "reference_generated_samples.pt")
            fake_array = fake_samples.numpy()

            axis.scatter(real_array[:, 0], real_array[:, 1], s=4, alpha=0.12, color="gray")
            axis.scatter(fake_array[:, 0], fake_array[:, 1], s=4, alpha=0.35, color="tab:blue")
            axis.set_xlim(x_min, x_max)
            axis.set_ylim(y_min, y_max)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(f"gp={gp_lambda:g}, n_critic={n_critic}")
            axis.text(
                0.03,
                0.97,
                f"W1={result['w1_mean']:.3f}\nvar_err={result['var_error_mean']:.3f}",
                transform=axis.transAxes,
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

            if row_index == len(gp_lambdas) - 1:
                axis.set_xlabel("x1")
            if col_index == 0:
                axis.set_ylabel("x2")

    figure.suptitle("Reference-seed evaluation samples in gray, WGAN-GP samples in blue")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_summary_heatmaps(aggregate_results, output_path):
    gp_lambdas = sorted({row["gp_lambda"] for row in aggregate_results})
    n_critics = sorted({row["n_critic"] for row in aggregate_results})
    results_map = {(row["gp_lambda"], row["n_critic"]): row for row in aggregate_results}

    metric_specs = [
        ("var_error", "Variance Error"),
        ("mean_error", "Mean Error"),
        ("fake_corr", "Abs Generated Corr"),
        ("w1", "Test W1"),
        ("w2", "Test W2"),
        ("selected_epoch", "Selected Epoch"),
    ]

    figure, axes = plt.subplots(2, 3, figsize=(15, 8))

    for axis, (metric, title) in zip(axes.flatten(), metric_specs):
        values = []
        labels = []
        for gp_lambda in gp_lambdas:
            value_row = []
            label_row = []
            for n_critic in n_critics:
                result = results_map[(gp_lambda, n_critic)]
                mean_value = result[f"{metric}_mean"]
                std_value = result[f"{metric}_std"]
                if metric == "fake_corr":
                    mean_value = abs(mean_value)
                value_row.append(mean_value)
                label_row.append(f"{mean_value:.3f}\n±{std_value:.3f}")
            values.append(value_row)
            labels.append(label_row)

        image = axis.imshow(values, cmap="viridis", aspect="auto")
        axis.set_title(title)
        axis.set_xlabel("n_critic")
        axis.set_ylabel("gp_lambda")
        axis.set_xticks(range(len(n_critics)), labels=[str(value) for value in n_critics])
        axis.set_yticks(range(len(gp_lambdas)), labels=[f"{value:g}" for value in gp_lambdas])

        for row_index, row in enumerate(labels):
            for col_index, value in enumerate(row):
                axis.text(col_index, row_index, value, ha="center", va="center", color="white", fontsize=8)

        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle("WGAN-GP Hyperparameter Sweep Summary")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_text_summary(aggregate_results, output_path):
    lines = []

    for row in sorted(aggregate_results, key=lambda result: result["w1_mean"]):
        lines.extend(
            [
                f"gp_lambda = {row['gp_lambda']:.6g}, n_critic = {row['n_critic']}",
                f"  best_val_w1: {row['best_val_w1_mean']:.4f} +/- {row['best_val_w1_std']:.4f}",
                f"  selected_val_w1: {row['selected_val_w1_mean']:.4f} +/- {row['selected_val_w1_std']:.4f}",
                f"  test_w1: {row['w1_mean']:.4f} +/- {row['w1_std']:.4f}",
                f"  test_w2: {row['w2_mean']:.4f} +/- {row['w2_std']:.4f}",
                f"  mean_error: {row['mean_error_mean']:.4f} +/- {row['mean_error_std']:.4f}",
                f"  var_error: {row['var_error_mean']:.4f} +/- {row['var_error_std']:.4f}",
                f"  fake_corr: {row['fake_corr_mean']:.4f} +/- {row['fake_corr_std']:.4f}",
                f"  selected_epoch: {row['selected_epoch_mean']:.1f} +/- {row['selected_epoch_std']:.1f}",
                f"  num_seeds: {row['num_seeds']}",
                "",
            ]
        )

    output_path.write_text("\n".join(lines).strip() + "\n")


def run_single_seed(args, gp_lambda, n_critic, seed, device):
    torch.manual_seed(seed)
    train_samples = make_correlated_gaussian_samples(args.n_samples, args.rho, seed)
    val_samples = make_correlated_gaussian_samples(args.val_samples, args.rho, seed + 1)
    eval_samples = make_correlated_gaussian_samples(args.eval_samples, args.rho, seed + 2)
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
    lr_g, lr_c = resolve_learning_rates(args.lr, args.generator_lr, args.critic_lr)
    trainer = WGANTrainer(
        generator=generator,
        critic=critic,
        device=device,
        lr_g=lr_g,
        lr_c=lr_c,
        n_critic=n_critic,
        gp_lambda=gp_lambda,
        optimizer_name=args.optimizer,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
    )

    history = trainer.fit(
        dataloader,
        args.num_epochs,
        eval_metric_samples=val_samples if args.w1_eval_period > 0 else None,
        metric_period=args.w1_eval_period,
        metric_max_samples=args.ot_eval_samples,
        metric_seed=seed,
        checkpoint_selection="best_eval_w1" if args.checkpoint_selection == "best_val_w1" else "last",
    )
    fake_samples = trainer.sample(args.eval_samples, seed=seed + 3).cpu()
    metrics = compute_metrics(eval_samples, fake_samples, history, args.ot_eval_samples, seed)

    return {
        "history": history,
        "eval_samples": eval_samples,
        "fake_samples": fake_samples,
        "metrics": metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep WGAN-GP hyperparameters (gp_lambda and n_critic) on the 2D Gaussian task."
    )
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=2000)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--rho", type=float, default=0.8)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/wgan_gp_hparam_sweep"))
    parser.add_argument("--gp-lambdas", type=float, nargs="+", default=[2.0, 5.0, 10.0, 20.0])
    parser.add_argument("--n-critics", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--ot-eval-samples", type=int, default=512)
    parser.add_argument("--w1-eval-period", type=int, default=10)
    parser.add_argument("--checkpoint-selection", choices=["last", "best_val_w1"], default="best_val_w1")
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    if args.num_seeds < 1:
        raise ValueError(f"Expected num_seeds to be at least 1, got {args.num_seeds}")

    seeds = args.seeds if args.seeds else list(range(args.seed, args.seed + args.num_seeds))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_results = []
    reference_seed = seeds[0]
    reference_eval_samples = None
    total_runs = len(args.gp_lambdas) * len(args.n_critics) * len(seeds)
    run_index = 0

    for gp_lambda in args.gp_lambdas:
        for n_critic in args.n_critics:
            run_dir = args.output_dir / f"gp_{gp_lambda:g}_ncritic_{n_critic}"
            run_dir.mkdir(parents=True, exist_ok=True)

            for seed in seeds:
                run_index += 1
                print(f"[{run_index}/{total_runs}] Running gp_lambda={gp_lambda:g}, n_critic={n_critic}, seed={seed}")
                run = run_single_seed(args, gp_lambda, n_critic, seed, device)
                metrics = run["metrics"]
                metrics["gp_lambda"] = gp_lambda
                metrics["n_critic"] = n_critic
                metrics["seed"] = seed
                metrics["run_dir"] = str(run_dir)
                per_seed_results.append(metrics)

                if seed == reference_seed:
                    reference_eval_samples = run["eval_samples"]
                    save_loss_plot(run["history"], run_dir / "reference_wgan_losses.png")
                    save_scatter_plot(run["eval_samples"], run["fake_samples"], run_dir / "reference_scatter.png")
                    save_marginal_plot(run["eval_samples"], run["fake_samples"], run_dir / "reference_marginals.png")
                    summarize_samples(
                        run["eval_samples"],
                        run["fake_samples"],
                        run_dir / "reference_summary.txt",
                        extra_metrics={
                            "Best validation W1": metrics["best_val_w1"],
                            "Test W1": metrics["w1"],
                            "Test W2": metrics["w2"],
                        },
                    )
                    torch.save(run["history"], run_dir / "reference_history.pt")
                    torch.save(run["fake_samples"], run_dir / "reference_generated_samples.pt")

                print(
                    f"[{run_index}/{total_runs}] Finished gp_lambda={gp_lambda:g}, n_critic={n_critic}, seed={seed} | "
                    f"best_val_w1={metrics['best_val_w1']:.4f} | "
                    f"test_w1={metrics['w1']:.4f} | "
                    f"test_w2={metrics['w2']:.4f} | "
                    f"selected_epoch={metrics['selected_epoch']}"
                )

    aggregate = aggregate_results(per_seed_results)
    save_results_csv(per_seed_results, args.output_dir / "per_seed_results.csv")
    save_results_csv(aggregate, args.output_dir / "aggregate_results.csv")
    save_summary_heatmaps(aggregate, args.output_dir / "summary_heatmaps.png")
    save_text_summary(aggregate, args.output_dir / "summary.txt")

    if reference_eval_samples is not None:
        torch.save(reference_eval_samples, args.output_dir / "reference_eval_samples.pt")
        save_reference_grid(reference_eval_samples, aggregate, args.output_dir / "reference_sweep_grid.png")

    print("Aggregate results:")
    for row in sorted(aggregate, key=lambda result: result["w1_mean"]):
        print(
            f"gp_lambda={row['gp_lambda']:.6g}, n_critic={row['n_critic']}: "
            f"best_val_w1={row['best_val_w1_mean']:.4f} +/- {row['best_val_w1_std']:.4f}, "
            f"test_w1={row['w1_mean']:.4f} +/- {row['w1_std']:.4f}, "
            f"test_w2={row['w2_mean']:.4f} +/- {row['w2_std']:.4f}, "
            f"var_error={row['var_error_mean']:.4f} +/- {row['var_error_std']:.4f}, "
            f"selected_epoch={row['selected_epoch_mean']:.1f} +/- {row['selected_epoch_std']:.1f}"
        )

    print(f"Finished. Sweep results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
