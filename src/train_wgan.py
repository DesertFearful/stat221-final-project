import torch
from torch.optim import RMSprop


class WGANTrainer:
    def __init__(self, generator, critic, device, lr=5e-5, n_critic=5, weight_clip=0.01):
        if n_critic < 1:
            raise ValueError(f"Expected n_critic to be at least 1, got {n_critic}")
        if weight_clip <= 0:
            raise ValueError(f"Expected weight_clip to be positive, got {weight_clip}")

        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.device = device
        self.lr = lr
        self.n_critic = n_critic
        self.weight_clip = weight_clip

        self.g_optimizer = RMSprop(self.generator.parameters(), lr=lr)
        self.c_optimizer = RMSprop(self.critic.parameters(), lr=lr)

        self.history = self._empty_history()
        self.step = 0

    def _empty_history(self):
        return {
            "generator_loss": [],
            "critic_loss": [],
            "real_score": [],
            "fake_score": [],
        }

    # sample independent latent variables for each output coordinate
    def sample_latent(self, batch_size):
        return torch.randn(batch_size, self.generator.data_dim, self.generator.latent_dim, device=self.device)

    # generate model samples
    def sample(self, batch_size):
        with torch.no_grad():
            z = self.sample_latent(batch_size)
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

    # full WGAN training loop
    def fit(self, dataloader, num_epochs):
        self.generator.train()
        self.critic.train()
        self.history = self._empty_history()
        self.step = 0

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

            self.history["generator_loss"].append(mean_generator_loss)
            self.history["critic_loss"].append(mean_critic_loss)
            self.history["real_score"].append(mean_real_score)
            self.history["fake_score"].append(mean_fake_score)

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"G: {mean_generator_loss:.4f} | "
                f"C: {mean_critic_loss:.4f}"
            )

        return self.history