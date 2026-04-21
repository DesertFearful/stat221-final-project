import math

import torch
from torch.optim import Adam, RMSprop

from src.ot_metrics import estimate_pot_wasserstein


class WGANTrainer:
    def __init__(
        self,
        generator,
        critic,
        device,
        lr=1e-4,
        n_critic=5,
        weight_clip=0.05,
        optimizer_name="rmsprop",
        adam_beta1=0.0,
        adam_beta2=0.9,
    ):
        if n_critic < 1:
            raise ValueError(f"Expected n_critic to be at least 1, got {n_critic}")
        if weight_clip <= 0:
            raise ValueError(f"Expected weight_clip to be positive, got {weight_clip}")
        if optimizer_name not in {"rmsprop", "adam"}:
            raise ValueError(f"Expected optimizer_name to be 'rmsprop' or 'adam', got {optimizer_name}")
        if not 0.0 <= adam_beta1 < 1.0:
            raise ValueError(f"Expected adam_beta1 to lie in [0, 1), got {adam_beta1}")
        if not 0.0 <= adam_beta2 < 1.0:
            raise ValueError(f"Expected adam_beta2 to lie in [0, 1), got {adam_beta2}")

        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.device = device
        self.lr = lr
        self.n_critic = n_critic
        self.weight_clip = weight_clip
        self.optimizer_name = optimizer_name
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2

        self.g_optimizer = self.make_optimizer(self.generator.parameters())
        self.c_optimizer = self.make_optimizer(self.critic.parameters())

        self.history = self._empty_history()
        self.step = 0

    def make_optimizer(self, parameters):
        if self.optimizer_name == "rmsprop":
            return RMSprop(parameters, lr=self.lr)

        return Adam(parameters, lr=self.lr, betas=(self.adam_beta1, self.adam_beta2))

    def _empty_history(self):
        return {
            "generator_loss": [],
            "critic_loss": [],
            "real_score": [],
            "fake_score": [],
            "train_w1": [],
            "eval_w1": [],
            "optimizer_name": self.optimizer_name,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "best_epoch": None,
            "best_eval_w1": float("nan"),
            "selected_epoch": None,
            "selected_eval_w1": float("nan"),
            "checkpoint_selection": "last",
        }

    # sample independent latent variables for each output coordinate
    def sample_latent(self, batch_size, seed=None):
        if seed is None:
            return torch.randn(batch_size, self.generator.data_dim, self.generator.latent_dim, device=self.device)

        generator = torch.Generator().manual_seed(seed)
        z = torch.randn(batch_size, self.generator.data_dim, self.generator.latent_dim, generator=generator)
        return z.to(self.device)

    # generate model samples
    def sample(self, batch_size, seed=None):
        with torch.no_grad():
            z = self.sample_latent(batch_size, seed=seed)
            return self.generator(z)

    def critic_step(self, real_batch, fake_batch):
        self.c_optimizer.zero_grad(set_to_none=True)

        real_score = self.critic(real_batch)
        fake_score = self.critic(fake_batch)

        loss = fake_score.mean() - real_score.mean()
        loss.backward()
        self.c_optimizer.step()

        with torch.no_grad():
            for parameter in self.critic.parameters():
                parameter.clamp_(-self.weight_clip, self.weight_clip)

        return loss.item(), real_score.mean().item(), fake_score.mean().item()

    def generator_step(self, batch_size):
        self.g_optimizer.zero_grad(set_to_none=True)

        z = self.sample_latent(batch_size)
        fake_batch = self.generator(z)
        loss = -self.critic(fake_batch).mean()

        loss.backward()
        self.g_optimizer.step()

        return loss.item()

    # evaluate the current generator against a fixed reference sample set
    def estimate_w1(self, real_samples, max_samples=512, seed=0):
        real_samples = real_samples.detach().cpu()
        fake_samples = self.sample(real_samples.shape[0], seed=seed).detach().cpu()
        metrics = estimate_pot_wasserstein(real_samples, fake_samples, max_samples=max_samples, seed=seed)
        return metrics["w1"]

    # keep the best generator parameters on CPU so they can be restored after training
    def copy_generator_state(self):
        return {name: tensor.detach().cpu().clone() for name, tensor in self.generator.state_dict().items()}

    # full WGAN training loop
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
    ):
        if metric_period < 0:
            raise ValueError(f"Expected metric_period to be nonnegative, got {metric_period}")
        if checkpoint_selection not in {"last", "best_eval_w1"}:
            raise ValueError(
                f"Expected checkpoint_selection to be one of ('last', 'best_eval_w1'), got {checkpoint_selection}"
            )

        track_w1 = train_metric_samples is not None or eval_metric_samples is not None
        if track_w1 and metric_period < 1:
            raise ValueError("metric_period must be at least 1 when train_metric_samples or eval_metric_samples are given.")
        if checkpoint_selection == "best_eval_w1" and eval_metric_samples is None:
            raise ValueError("eval_metric_samples must be provided when checkpoint_selection='best_eval_w1'.")

        self.generator.train()
        self.critic.train()
        self.history = self._empty_history()
        self.history["checkpoint_selection"] = checkpoint_selection
        self.step = 0
        best_generator_state = None
        best_epoch = None
        best_eval_w1 = float("inf")

        for epoch in range(num_epochs):
            epoch_generator_losses = []
            epoch_critic_losses = []
            epoch_real_scores = []
            epoch_fake_scores = []

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
                critic_loss, real_score, fake_score = self.critic_step(real_batch, fake_batch)

                epoch_critic_losses.append(critic_loss)
                epoch_real_scores.append(real_score)
                epoch_fake_scores.append(fake_score)

                if (self.step + 1) % self.n_critic == 0:
                    generator_loss = self.generator_step(real_batch.shape[0])
                    epoch_generator_losses.append(generator_loss)

                self.step += 1

            if epoch_generator_losses:
                mean_generator_loss = sum(epoch_generator_losses) / len(epoch_generator_losses)
            else:
                mean_generator_loss = float("nan")

            mean_critic_loss = sum(epoch_critic_losses) / len(epoch_critic_losses)
            mean_real_score = sum(epoch_real_scores) / len(epoch_real_scores)
            mean_fake_score = sum(epoch_fake_scores) / len(epoch_fake_scores)
            train_w1 = float("nan")
            eval_w1 = float("nan")

            if track_w1 and (epoch + 1) % metric_period == 0:
                train_w1_seed = metric_seed
                eval_w1_seed = metric_seed + 1
                self.generator.eval()

                if train_metric_samples is not None:
                    train_w1 = self.estimate_w1(train_metric_samples, max_samples=metric_max_samples, seed=train_w1_seed)
                if eval_metric_samples is not None:
                    eval_w1 = self.estimate_w1(eval_metric_samples, max_samples=metric_max_samples, seed=eval_w1_seed)
                    if eval_w1 < best_eval_w1:
                        best_eval_w1 = eval_w1
                        best_epoch = epoch + 1
                        best_generator_state = self.copy_generator_state()

                self.generator.train()

            self.history["generator_loss"].append(mean_generator_loss)
            self.history["critic_loss"].append(mean_critic_loss)
            self.history["real_score"].append(mean_real_score)
            self.history["fake_score"].append(mean_fake_score)
            self.history["train_w1"].append(train_w1)
            self.history["eval_w1"].append(eval_w1)

            message = (
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"G: {mean_generator_loss:.4f} | "
                f"C: {mean_critic_loss:.4f}"
            )
            if not math.isnan(train_w1):
                message += f" | train W1: {train_w1:.4f}"
            if not math.isnan(eval_w1):
                message += f" | eval W1: {eval_w1:.4f}"

            print(message)

        if best_epoch is not None:
            self.history["best_epoch"] = best_epoch
            self.history["best_eval_w1"] = best_eval_w1

        if checkpoint_selection == "best_eval_w1":
            if best_generator_state is None:
                raise ValueError(
                    "No validation W1 was recorded during training. Reduce metric_period or increase num_epochs."
                )

            self.generator.load_state_dict(best_generator_state)
            self.history["selected_epoch"] = best_epoch
            self.history["selected_eval_w1"] = best_eval_w1
            print(f"Selected checkpoint from epoch {best_epoch} with validation W1 {best_eval_w1:.4f}")
        else:
            self.history["selected_epoch"] = num_epochs
            if self.history["eval_w1"]:
                eval_values = [value for value in self.history["eval_w1"] if not math.isnan(value)]
                if eval_values:
                    self.history["selected_eval_w1"] = eval_values[-1]

        return self.history
