import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import Critic, FactorizedGenerator
from src.train_wgan import WGANTrainer


DEFAULT_NUM_EPOCHS = 400
DEFAULT_GENERATOR_LR = 1e-4
DEFAULT_CRITIC_LR = 6e-4
DEFAULT_N_CRITIC = 5
DEFAULT_GP_LAMBDA = 1.0
DEFAULT_CRITIC_FEATURE_MAP = "raw"


# Choose a usable device while keeping the command-line interface simple.
def resolve_device(device_name):
    mps_is_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if device_name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if mps_is_available:
            return "mps"
        return "cpu"

    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available on this machine.")
    if device_name == "mps" and not mps_is_available:
        raise ValueError("MPS was requested but is not available on this machine.")

    return device_name


def build_hidden_dims(width, depth):
    if depth < 0:
        raise ValueError(f"Expected depth to be nonnegative, got {depth}")
    if width < 1:
        raise ValueError(f"Expected width to be positive, got {width}")
    return tuple(width for _ in range(depth))


def resolve_learning_rates(shared_lr, generator_lr, critic_lr):
    if shared_lr is not None:
        if generator_lr is not None or critic_lr is not None:
            raise ValueError("Use either --lr or the pair --generator-lr/--critic-lr, not both.")
        return shared_lr, shared_lr

    lr_g = DEFAULT_GENERATOR_LR if generator_lr is None else generator_lr
    lr_c = DEFAULT_CRITIC_LR if critic_lr is None else critic_lr
    return lr_g, lr_c


# Sample from a centered correlated Gaussian in R^2.
def make_correlated_gaussian_samples(n_samples, rho, seed):
    if n_samples < 1:
        raise ValueError(f"Expected n_samples to be positive, got {n_samples}")
    if abs(rho) >= 1:
        raise ValueError(f"Expected |rho| < 1, got {rho}")

    covariance = torch.tensor([[1.0, rho], [rho, 1.0]], dtype=torch.float32)
    chol = torch.linalg.cholesky(covariance)
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(n_samples, 2, generator=generator)
    return z @ chol.T


# Wrap the synthetic samples in a shuffled dataloader.
def make_dataloader(samples, batch_size, seed=None):
    dataset = TensorDataset(samples)

    if seed is None:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


# Plot the real and generated point clouds side by side.
def save_scatter_plot(real_samples, fake_samples, output_path):
    real_array = real_samples.numpy()
    fake_array = fake_samples.numpy()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(real_array[:, 0], real_array[:, 1], s=8, alpha=0.5)
    axes[0].set_title("Real Samples")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")

    axes[1].scatter(fake_array[:, 0], fake_array[:, 1], s=8, alpha=0.5)
    axes[1].set_title("Generated Samples")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")

    x_min = min(real_array[:, 0].min(), fake_array[:, 0].min())
    x_max = max(real_array[:, 0].max(), fake_array[:, 0].max())
    y_min = min(real_array[:, 1].min(), fake_array[:, 1].min())
    y_max = max(real_array[:, 1].max(), fake_array[:, 1].max())

    for axis in axes:
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_aspect("equal", adjustable="box")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


# Compare the coordinate marginals of the real and generated samples.
def save_marginal_plot(real_samples, fake_samples, output_path):
    real_array = real_samples.numpy()
    fake_array = fake_samples.numpy()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    for index, axis in enumerate(axes):
        axis.hist(real_array[:, index], bins=40, density=True, alpha=0.6, label="real")
        axis.hist(fake_array[:, index], bins=40, density=True, alpha=0.6, label="generated")
        axis.set_title(f"Marginal x{index + 1}")
        axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


# Save the generator and critic losses across training epochs.
def save_loss_plot(history, output_path):
    has_w1 = any(not math.isnan(value) for value in history.get("train_w1", []))
    has_w1 = has_w1 or any(not math.isnan(value) for value in history.get("eval_w1", []))
    epochs = range(1, len(history["generator_loss"]) + 1)

    if has_w1:
        figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        loss_axis, w1_axis = axes
    else:
        figure, loss_axis = plt.subplots(figsize=(8, 4))
        w1_axis = None

    loss_axis.plot(epochs, history["generator_loss"], label="generator")
    loss_axis.plot(epochs, history["critic_loss"], label="critic")
    if "gradient_penalty" in history:
        loss_axis.plot(epochs, history["gradient_penalty"], label="gradient penalty")
    selected_epoch = None
    if history.get("checkpoint_selection") == "best_eval_w1":
        selected_epoch = history.get("selected_epoch")
    if selected_epoch is not None:
        loss_axis.axvline(selected_epoch, color="black", linestyle="--", alpha=0.5, label="selected checkpoint")
    loss_axis.set_title("Training Losses")
    loss_axis.set_xlabel("Epoch")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()

    if w1_axis is not None:
        train_points = [
            (epoch, value) for epoch, value in zip(epochs, history["train_w1"]) if not math.isnan(value)
        ]
        eval_points = [
            (epoch, value) for epoch, value in zip(epochs, history["eval_w1"]) if not math.isnan(value)
        ]

        if train_points:
            train_epochs, train_values = zip(*train_points)
            w1_axis.plot(train_epochs, train_values, marker="o", label="train W1")
        if eval_points:
            eval_epochs, eval_values = zip(*eval_points)
            w1_axis.plot(eval_epochs, eval_values, marker="o", label="eval W1")
        if selected_epoch is not None:
            w1_axis.axvline(selected_epoch, color="black", linestyle="--", alpha=0.5, label="selected checkpoint")

        w1_axis.set_title("W1 Over Training")
        w1_axis.set_xlabel("Epoch")
        w1_axis.grid(alpha=0.25)
        w1_axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


# Summarize the first and second moments of the real and generated samples.
def summarize_samples(real_samples, fake_samples, output_path, extra_metrics=None):
    real_mean = real_samples.mean(dim=0)
    fake_mean = fake_samples.mean(dim=0)
    real_cov = torch.cov(real_samples.T)
    fake_cov = torch.cov(fake_samples.T)
    real_corr = torch.corrcoef(real_samples.T)[0, 1]
    fake_corr = torch.corrcoef(fake_samples.T)[0, 1]

    lines = [
        "Real mean:",
        str(real_mean.tolist()),
        "",
        "Generated mean:",
        str(fake_mean.tolist()),
        "",
        "Real covariance:",
        str(real_cov.tolist()),
        "",
        "Generated covariance:",
        str(fake_cov.tolist()),
        "",
        f"Real correlation: {real_corr.item():.4f}",
        f"Generated correlation: {fake_corr.item():.4f}",
    ]

    if extra_metrics:
        lines.extend([""])
        for name, value in extra_metrics.items():
            lines.append(f"{name}: {value:.4f}")

    output_path.write_text("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Train a WGAN on a correlated 2D Gaussian.")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--rho", type=float, default=0.8)
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
    parser.add_argument("--w1-eval-period", type=int, default=10)
    parser.add_argument("--w1-eval-samples", type=int, default=512)
    parser.add_argument("--checkpoint-selection", choices=["last", "best_val_w1"], default="best_val_w1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gaussian_2d"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    lr_g, lr_c = resolve_learning_rates(args.lr, args.generator_lr, args.critic_lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    real_samples = make_correlated_gaussian_samples(args.n_samples, args.rho, args.seed)
    val_samples = make_correlated_gaussian_samples(args.n_samples, args.rho, args.seed + 1)
    dataloader = make_dataloader(real_samples, args.batch_size)

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
        n_critic=args.n_critic,
        gp_lambda=args.gp_lambda,
        optimizer_name=args.optimizer,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
    )

    history = trainer.fit(
        dataloader,
        args.num_epochs,
        train_metric_samples=real_samples if args.w1_eval_period > 0 else None,
        eval_metric_samples=val_samples if args.w1_eval_period > 0 else None,
        metric_period=args.w1_eval_period,
        metric_max_samples=args.w1_eval_samples,
        metric_seed=args.seed,
        checkpoint_selection="best_eval_w1" if args.checkpoint_selection == "best_val_w1" else "last",
    )
    fake_samples = trainer.sample(args.n_samples).cpu()
    extra_metrics = {}

    if args.w1_eval_period > 0:
        selected_train_w1 = trainer.estimate_w1(real_samples, max_samples=args.w1_eval_samples, seed=args.seed)
        selected_val_w1 = trainer.estimate_w1(val_samples, max_samples=args.w1_eval_samples, seed=args.seed + 1)
        extra_metrics["Selected train W1"] = selected_train_w1
        extra_metrics["Selected validation W1"] = selected_val_w1

    if history.get("selected_epoch") is not None:
        extra_metrics["Selected epoch"] = float(history["selected_epoch"])

    if math.isnan(history.get("best_eval_w1", float("nan"))) is False:
        extra_metrics["Best validation W1"] = history["best_eval_w1"]

    if not extra_metrics:
        extra_metrics = None

    torch.save(real_samples, args.output_dir / "real_samples.pt")
    torch.save(fake_samples, args.output_dir / "generated_samples.pt")
    save_scatter_plot(real_samples, fake_samples, args.output_dir / "scatter.png")
    save_marginal_plot(real_samples, fake_samples, args.output_dir / "marginals.png")
    save_loss_plot(history, args.output_dir / "losses.png")
    summarize_samples(real_samples, fake_samples, args.output_dir / "summary.txt", extra_metrics=extra_metrics)

    print(f"Finished. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
