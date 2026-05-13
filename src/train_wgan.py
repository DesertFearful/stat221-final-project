import math

import torch
from torch.autograd import grad
from torch.optim import Adam, RMSprop

from src.ot_metrics import estimate_pot_wasserstein


class WGANTrainer:
    def __init__(
        self,
        generator,
        critic,
        device,
        lr=None,
        lr_g=None,
        lr_c=None,
        n_critic=5,
        gp_lambda=10.0,
        optimizer_name="adam",
        adam_beta1=0.0,
        adam_beta2=0.9,
    ):
        if n_critic < 1:
            raise ValueError(f"Expected n_critic to be at least 1, got {n_critic}")
        if gp_lambda < 0:
            raise ValueError(f"Expected gp_lambda to be nonnegative, got {gp_lambda}")
        if optimizer_name not in {"rmsprop", "adam"}:
            raise ValueError(f"Expected optimizer_name to be 'rmsprop' or 'adam', got {optimizer_name}")
        if not 0.0 <= adam_beta1 < 1.0:
            raise ValueError(f"Expected adam_beta1 to lie in [0, 1), got {adam_beta1}")
        if not 0.0 <= adam_beta2 < 1.0:
            raise ValueError(f"Expected adam_beta2 to lie in [0, 1), got {adam_beta2}")

        if lr is None and lr_g is None and lr_c is None:
            lr = 1e-4
        self.lr_g = lr if lr_g is None else lr_g
        self.lr_c = lr if lr_c is None else lr_c
        if self.lr_g <= 0 or self.lr_c <= 0:
            raise ValueError(f"Expected positive learning rates, got lr_g={self.lr_g}, lr_c={self.lr_c}")

        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.device = device
        self.n_critic = n_critic
        self.gp_lambda = gp_lambda
        self.optimizer_name = optimizer_name
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2

        self.g_optimizer = self.make_optimizer(self.generator.parameters(), lr_value=self.lr_g)
        self.c_optimizer = self.make_optimizer(self.critic.parameters(), lr_value=self.lr_c)

        self.history = self._empty_history()
        self.step = 0

    def make_optimizer(self, parameters, lr_value):
        if self.optimizer_name == "rmsprop":
            return RMSprop(parameters, lr=lr_value)

        return Adam(parameters, lr=lr_value, betas=(self.adam_beta1, self.adam_beta2))

    def _empty_history(self):
        return {
            "generator_loss": [],
            "critic_loss": [],
            "critic_objective": [],
            "gradient_penalty": [],
            "real_score": [],
            "fake_score": [],
            "train_w1": [],
            "eval_w1": [],
            "eval_mean_error": [],
            "eval_diag_var_error": [],
            "eval_checkpoint_score": [],
            "optimizer_name": self.optimizer_name,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "lr_g": self.lr_g,
            "lr_c": self.lr_c,
            "gp_lambda": self.gp_lambda,
            "best_epoch": None,
            "best_eval_w1": float("nan"),
            "best_eval_mean_error": float("nan"),
            "best_eval_diag_var_error": float("nan"),
            "best_checkpoint_score": float("nan"),
            "selected_epoch": None,
            "selected_eval_w1": float("nan"),
            "selected_eval_mean_error": float("nan"),
            "selected_eval_diag_var_error": float("nan"),
            "selected_checkpoint_score": float("nan"),
            "checkpoint_selection": "last",
            "checkpoint_var_weight": 0.0,
            "checkpoint_mean_weight": 0.0,
        }

    def sample_latent(self, batch_size, seed=None):
        if seed is None:
            return torch.randn(batch_size, self.generator.data_dim, self.generator.latent_dim, device=self.device)

        generator = torch.Generator().manual_seed(seed)
        z = torch.randn(batch_size, self.generator.data_dim, self.generator.latent_dim, generator=generator)
        return z.to(self.device)

    def sample(self, batch_size, seed=None):
        with torch.no_grad():
            z = self.sample_latent(batch_size, seed=seed)
            return self.generator(z)

    def gradient_penalty(self, real_batch, fake_batch):
        batch_size = real_batch.shape[0]
        alpha = torch.rand(batch_size, 1, device=self.device)
        alpha = alpha.expand_as(real_batch)
        interpolated = alpha * real_batch + (1.0 - alpha) * fake_batch
        interpolated.requires_grad_(True)

        scores = self.critic(interpolated)
        gradients = grad(
            outputs=scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(batch_size, -1)
        return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()

    def critic_step(self, real_batch, fake_batch):
        self.c_optimizer.zero_grad(set_to_none=True)

        real_score = self.critic(real_batch)
        fake_score = self.critic(fake_batch)
        objective = fake_score.mean() - real_score.mean()
        gp = self.gradient_penalty(real_batch, fake_batch) if self.gp_lambda > 0 else torch.tensor(0.0, device=self.device)
        loss = objective + self.gp_lambda * gp

        loss.backward()
        self.c_optimizer.step()

        return (
            loss.item(),
            real_score.mean().item(),
            fake_score.mean().item(),
            gp.item(),
            objective.item(),
        )

    def generator_step(self, batch_size):
        self.g_optimizer.zero_grad(set_to_none=True)

        z = self.sample_latent(batch_size)
        fake_batch = self.generator(z)
        loss = -self.critic(fake_batch).mean()

        loss.backward()
        self.g_optimizer.step()

        return loss.item()

    def estimate_w1(self, real_samples, max_samples=512, seed=0):
        real_samples = real_samples.detach().cpu()
        fake_samples = self.sample(real_samples.shape[0], seed=seed).detach().cpu()
        metrics = estimate_pot_wasserstein(real_samples, fake_samples, max_samples=max_samples, seed=seed)
        return metrics["w1"]

    def estimate_eval_metrics(self, real_samples, max_samples=512, seed=0):
        real_samples = real_samples.detach().cpu()
        fake_samples = self.sample(real_samples.shape[0], seed=seed).detach().cpu()
        ot_metrics = estimate_pot_wasserstein(real_samples, fake_samples, max_samples=max_samples, seed=seed)

        real_mean = real_samples.mean(dim=0)
        fake_mean = fake_samples.mean(dim=0)
        real_covariance = torch.cov(real_samples.T)
        fake_covariance = torch.cov(fake_samples.T)
        real_diag = torch.diag(real_covariance)
        fake_diag = torch.diag(fake_covariance)

        return {
            "w1": ot_metrics["w1"],
            "mean_error": torch.norm(fake_mean - real_mean).item(),
            "diag_var_error": (fake_diag - real_diag).abs().mean().item(),
        }

    def copy_generator_state(self):
        return {name: tensor.detach().cpu().clone() for name, tensor in self.generator.state_dict().items()}

    def fit(
        self,
        dataloader,
        num_epochs,
        train_metric_samples=None,
        eval_metric_samples=None,
        metric_period=0,
        metric_max_samples=512,
        metric_seed=0,
        checkpoint_selection="last",
        checkpoint_var_weight=0.0,
        checkpoint_mean_weight=0.0,
    ):
        if metric_period < 0:
            raise ValueError(f"Expected metric_period to be nonnegative, got {metric_period}")
        if checkpoint_selection not in {"last", "best_eval_w1", "best_eval_composite"}:
            raise ValueError(
                "Expected checkpoint_selection to be one of "
                f"('last', 'best_eval_w1', 'best_eval_composite'), got {checkpoint_selection}"
            )
        if checkpoint_var_weight < 0:
            raise ValueError(f"Expected checkpoint_var_weight to be nonnegative, got {checkpoint_var_weight}")
        if checkpoint_mean_weight < 0:
            raise ValueError(f"Expected checkpoint_mean_weight to be nonnegative, got {checkpoint_mean_weight}")

        track_w1 = train_metric_samples is not None or eval_metric_samples is not None
        if track_w1 and metric_period < 1:
            raise ValueError("metric_period must be at least 1 when train_metric_samples or eval_metric_samples are given.")
        if checkpoint_selection != "last" and eval_metric_samples is None:
            raise ValueError(
                f"eval_metric_samples must be provided when checkpoint_selection='{checkpoint_selection}'."
            )

        self.generator.train()
        self.critic.train()
        self.history = self._empty_history()
        self.history["checkpoint_selection"] = checkpoint_selection
        self.history["checkpoint_var_weight"] = checkpoint_var_weight
        self.history["checkpoint_mean_weight"] = checkpoint_mean_weight
        self.step = 0
        best_generator_state = None
        best_epoch = None
        best_eval_w1 = float("nan")
        best_eval_mean_error = float("nan")
        best_eval_diag_var_error = float("nan")
        best_checkpoint_score = float("inf")

        for epoch in range(num_epochs):
            epoch_generator_losses = []
            epoch_critic_losses = []
            epoch_real_scores = []
            epoch_fake_scores = []
            epoch_gp_losses = []
            epoch_critic_objectives = []

            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    real_batch = batch[0]
                else:
                    real_batch = batch

                if not isinstance(real_batch, torch.Tensor):
                    raise TypeError(
                        f"Expected dataloader to return a tensor or a tuple/list whose "
                        f"first element is a tensor, got {type(real_batch)}"
                    )

                real_batch = real_batch.to(self.device)
                fake_batch = self.sample(real_batch.shape[0]).detach()
                critic_loss, real_score, fake_score, gp_loss, critic_objective = self.critic_step(real_batch, fake_batch)

                epoch_critic_losses.append(critic_loss)
                epoch_real_scores.append(real_score)
                epoch_fake_scores.append(fake_score)
                epoch_gp_losses.append(gp_loss)
                epoch_critic_objectives.append(critic_objective)

                if (self.step + 1) % self.n_critic == 0:
                    generator_loss = self.generator_step(real_batch.shape[0])
                    epoch_generator_losses.append(generator_loss)

                self.step += 1

            mean_generator_loss = (
                sum(epoch_generator_losses) / len(epoch_generator_losses)
                if epoch_generator_losses
                else float("nan")
            )
            mean_critic_loss = sum(epoch_critic_losses) / len(epoch_critic_losses)
            mean_real_score = sum(epoch_real_scores) / len(epoch_real_scores)
            mean_fake_score = sum(epoch_fake_scores) / len(epoch_fake_scores)
            mean_gp_loss = sum(epoch_gp_losses) / len(epoch_gp_losses)
            mean_critic_objective = sum(epoch_critic_objectives) / len(epoch_critic_objectives)
            train_w1 = float("nan")
            eval_w1 = float("nan")
            eval_mean_error = float("nan")
            eval_diag_var_error = float("nan")
            eval_checkpoint_score = float("nan")

            if track_w1 and (epoch + 1) % metric_period == 0:
                train_w1_seed = metric_seed
                eval_w1_seed = metric_seed + 1
                self.generator.eval()

                if train_metric_samples is not None:
                    train_w1 = self.estimate_w1(train_metric_samples, max_samples=metric_max_samples, seed=train_w1_seed)
                if eval_metric_samples is not None:
                    eval_metrics = self.estimate_eval_metrics(
                        eval_metric_samples,
                        max_samples=metric_max_samples,
                        seed=eval_w1_seed,
                    )
                    eval_w1 = eval_metrics["w1"]
                    eval_mean_error = eval_metrics["mean_error"]
                    eval_diag_var_error = eval_metrics["diag_var_error"]
                    if checkpoint_selection == "best_eval_w1":
                        eval_checkpoint_score = eval_w1
                    elif checkpoint_selection == "best_eval_composite":
                        eval_checkpoint_score = (
                            eval_w1
                            + checkpoint_var_weight * eval_diag_var_error
                            + checkpoint_mean_weight * eval_mean_error
                        )

                    if checkpoint_selection != "last" and eval_checkpoint_score < best_checkpoint_score:
                        best_checkpoint_score = eval_checkpoint_score
                        best_eval_w1 = eval_w1
                        best_eval_mean_error = eval_mean_error
                        best_eval_diag_var_error = eval_diag_var_error
                        best_epoch = epoch + 1
                        best_generator_state = self.copy_generator_state()

                self.generator.train()

            self.history["generator_loss"].append(mean_generator_loss)
            self.history["critic_loss"].append(mean_critic_loss)
            self.history["critic_objective"].append(mean_critic_objective)
            self.history["gradient_penalty"].append(mean_gp_loss)
            self.history["real_score"].append(mean_real_score)
            self.history["fake_score"].append(mean_fake_score)
            self.history["train_w1"].append(train_w1)
            self.history["eval_w1"].append(eval_w1)
            self.history["eval_mean_error"].append(eval_mean_error)
            self.history["eval_diag_var_error"].append(eval_diag_var_error)
            self.history["eval_checkpoint_score"].append(eval_checkpoint_score)

            message = (
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"G: {mean_generator_loss:.4f} | "
                f"C: {mean_critic_loss:.4f} | "
                f"GP: {mean_gp_loss:.4f}"
            )
            if not math.isnan(train_w1):
                message += f" | train W1: {train_w1:.4f}"
            if not math.isnan(eval_w1):
                message += f" | eval W1: {eval_w1:.4f}"
            if not math.isnan(eval_checkpoint_score):
                message += (
                    f" | eval diag var err: {eval_diag_var_error:.4f}"
                    f" | checkpoint score: {eval_checkpoint_score:.4f}"
                )

            print(message)

        if best_epoch is not None:
            self.history["best_epoch"] = best_epoch
            self.history["best_eval_w1"] = best_eval_w1
            self.history["best_eval_mean_error"] = best_eval_mean_error
            self.history["best_eval_diag_var_error"] = best_eval_diag_var_error
            self.history["best_checkpoint_score"] = best_checkpoint_score

        if checkpoint_selection != "last":
            if best_generator_state is None:
                raise ValueError(
                    "No validation checkpoint score was recorded during training. "
                    "Reduce metric_period or increase num_epochs."
                )

            self.generator.load_state_dict(best_generator_state)
            self.history["selected_epoch"] = best_epoch
            self.history["selected_eval_w1"] = best_eval_w1
            self.history["selected_eval_mean_error"] = best_eval_mean_error
            self.history["selected_eval_diag_var_error"] = best_eval_diag_var_error
            self.history["selected_checkpoint_score"] = best_checkpoint_score
            print(
                f"Selected checkpoint from epoch {best_epoch} with validation W1 {best_eval_w1:.4f} "
                f"and checkpoint score {best_checkpoint_score:.4f}"
            )
        else:
            self.history["selected_epoch"] = num_epochs
            selected_fields = [
                ("eval_w1", "selected_eval_w1"),
                ("eval_mean_error", "selected_eval_mean_error"),
                ("eval_diag_var_error", "selected_eval_diag_var_error"),
                ("eval_checkpoint_score", "selected_checkpoint_score"),
            ]
            for source_field, selected_field in selected_fields:
                values = [value for value in self.history[source_field] if not math.isnan(value)]
                if values:
                    self.history[selected_field] = values[-1]

        return self.history
