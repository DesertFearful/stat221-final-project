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


def save_sweep_grid(real_samples, results, output_path):
    weight_clips = sorted({result["gp_lambda"] for result in results})
    n_critics = sorted({result["n_critic"] for result in results})
    results_map = {(result["gp_lambda"], result["n_critic"]): result for result in results}

    real_array = real_samples.numpy()
    x_min = real_array[:, 0].min()
    x_max = real_array[:, 0].max()
    y_min = real_array[:, 1].min()
    y_max = real_array[:, 1].max()

    figure, axes = plt.subplots(
        len(weight_clips),
        len(n_critics),
        figsize=(4 * len(n_critics), 4 * len(weight_clips)),
        sharex=True,
        sharey=True,
    )

    if len(weight_clips) == 1 and len(n_critics) == 1:
        axes = [[axes]]
    elif len(weight_clips) == 1:
        axes = [axes]
    elif len(n_critics) == 1:
        axes = [[axis] for axis in axes]

    for row_index, weight_clip in enumerate(weight_clips):
        for col_index, n_critic in enumerate(n_critics):
            axis = axes[row_index][col_index]
            result = results_map[(weight_clip, n_critic)]
            fake_samples = torch.load(Path(result["run_dir"]) / "generated_samples.pt")
            fake_array = fake_samples.numpy()

            axis.scatter(real_array[:, 0], real_array[:, 1], s=4, alpha=0.12, color="gray")
            axis.scatter(fake_array[:, 0], fake_array[:, 1], s=4, alpha=0.35, color="tab:blue")
            axis.set_xlim(x_min, x_max)
            axis.set_ylim(y_min, y_max)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(f"gp={weight_clip:g}, n_critic={n_critic}")
            axis.text(
                0.03,
                0.97,
                f"var=({result['fake_var_1']:.2f}, {result['fake_var_2']:.2f})\n"
                f"corr={result['fake_corr']:.3f}",
                transform=axis.transAxes,
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

            if row_index == len(weight_clips) - 1:
                axis.set_xlabel("x1")
            if col_index == 0:
                axis.set_ylabel("x2")

    figure.suptitle("WGAN-GP Sweep: real samples in gray, generated samples in blue")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_summary_heatmaps(results, output_path):
    weight_clips = sorted({result["gp_lambda"] for result in results})
    n_critics = sorted({result["n_critic"] for result in results})
    results_map = {(result["gp_lambda"], result["n_critic"]): result for result in results}

    var_error = []
    mean_error = []
    abs_corr = []
    w1_values = []
    w2_values = []

    for weight_clip in weight_clips:
        var_error_row = []
        mean_error_row = []
        abs_corr_row = []
        w1_row = []
        w2_row = []

        for n_critic in n_critics:
            result = results_map[(weight_clip, n_critic)]
            var_error_row.append(
                0.5 * (
                    abs(result["fake_var_1"] - result["real_var_1"]) +
                    abs(result["fake_var_2"] - result["real_var_2"])
                )
            )
            mean_error_row.append(
                (
                    (result["fake_mean_1"] - result["real_mean_1"]) ** 2 +
                    (result["fake_mean_2"] - result["real_mean_2"]) ** 2
                ) ** 0.5
            )
            abs_corr_row.append(abs(result["fake_corr"]))
            w1_row.append(result["w1"])
            w2_row.append(result["w2"])

        var_error.append(var_error_row)
        mean_error.append(mean_error_row)
        abs_corr.append(abs_corr_row)
        w1_values.append(w1_row)
        w2_values.append(w2_row)

    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    heatmaps = [
        (var_error, "Mean Abs Variance Error"),
        (mean_error, "Mean Error Norm"),
        (abs_corr, "Abs Generated Correlation"),
        (w1_values, "POT W1"),
        (w2_values, "POT W2"),
    ]

    flat_axes = axes.flatten()

    for axis, (values, title) in zip(flat_axes, heatmaps):
        image = axis.imshow(values, cmap="viridis", aspect="auto")
        axis.set_title(title)
        axis.set_xlabel("n_critic")
        axis.set_ylabel("gp_lambda")
        axis.set_xticks(range(len(n_critics)), labels=[str(n) for n in n_critics])
        axis.set_yticks(range(len(weight_clips)), labels=[f"{clip:g}" for clip in weight_clips])

        for row_index, row in enumerate(values):
            for col_index, value in enumerate(row):
                axis.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", color="white", fontsize=9)

        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    flat_axes[-1].axis("off")
    figure.suptitle("WGAN-GP Sweep Summary")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep gradient penalty and n_critic for the 2D Gaussian WGAN experiment.")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=2000)
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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gp_wgan_sweep"))
    parser.add_argument("--gp-lambdas", type=float, nargs="+", default=[2.0, 5.0, 10.0, 20.0])
    parser.add_argument("--n-critics", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--ot-eval-samples", type=int, default=512)
    parser.add_argument("--w1-eval-period", type=int, default=10)
    parser.add_argument("--checkpoint-selection", choices=["last", "best_val_w1"], default="best_val_w1")
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    lr_g, lr_c = resolve_learning_rates(args.lr, args.generator_lr, args.critic_lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    real_samples = make_correlated_gaussian_samples(args.n_samples, args.rho, args.seed)
    val_samples = make_correlated_gaussian_samples(args.val_samples, args.rho, args.seed + 1)
    torch.save(real_samples, args.output_dir / "real_samples.pt")

    results = []
    run_index = 0
    total_runs = len(args.gp_lambdas) * len(args.n_critics)

    for weight_clip in args.gp_lambdas:
        for n_critic in args.n_critics:
            run_index += 1
            run_name = f"gp_{weight_clip:g}_ncritic_{n_critic}"
            run_dir = args.output_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{run_index}/{total_runs}] Starting {run_name}")

            torch.manual_seed(args.seed)
            dataloader = make_dataloader(real_samples, args.batch_size, seed=args.seed)
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
            trainer = WGANTrainer(
                generator=generator,
                critic=critic,
                device=device,
                lr_g=lr_g,
                lr_c=lr_c,
                n_critic=n_critic,
                gp_lambda=weight_clip,
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
                metric_seed=args.seed,
                checkpoint_selection="best_eval_w1" if args.checkpoint_selection == "best_val_w1" else "last",
            )
            fake_samples = trainer.sample(args.n_samples).cpu()

            torch.save(fake_samples, run_dir / "generated_samples.pt")
            torch.save(history, run_dir / "history.pt")
            save_scatter_plot(real_samples, fake_samples, run_dir / "scatter.png")
            save_marginal_plot(real_samples, fake_samples, run_dir / "marginals.png")
            save_loss_plot(history, run_dir / "losses.png")
            metrics = compute_metrics(
                real_samples,
                fake_samples,
                history,
                ot_eval_samples=args.ot_eval_samples,
                ot_seed=args.seed,
            )
            summarize_samples(
                real_samples,
                fake_samples,
                run_dir / "summary.txt",
                extra_metrics={"POT W1": metrics["w1"], "POT W2": metrics["w2"]},
            )
            metrics["gp_lambda"] = weight_clip
            metrics["n_critic"] = n_critic
            metrics["run_dir"] = str(run_dir)
            results.append(metrics)

            print(
                f"[{run_index}/{total_runs}] Finished {run_name} | "
                f"fake_var=({metrics['fake_var_1']:.4f}, {metrics['fake_var_2']:.4f}) | "
                f"fake_corr={metrics['fake_corr']:.4f} | "
                f"W1={metrics['w1']:.4f} | "
                f"W2={metrics['w2']:.4f}"
            )

    save_results_csv(results, args.output_dir / "results.csv")
    save_sweep_grid(real_samples, results, args.output_dir / "sweep_grid.png")
    save_summary_heatmaps(results, args.output_dir / "summary_heatmaps.png")
    print(f"Finished. Sweep results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
