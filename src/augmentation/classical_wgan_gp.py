"""Classical WGAN-GP implementation (PyTorch) for tabular minority sample generation.

This is a compact, first-pass implementation intended for experimentation.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(_weights_init)

    def forward(self, z):
        return self.model(z)


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(_weights_init)

    def forward(self, x):
        return self.model(x).view(-1)


def _gradient_penalty(critic, real, fake, device='cpu'):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = critic(interpolates)
    grad_outputs = torch.ones_like(d_interpolates, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty


def generate_wgan_samples(
    X_minority,
    n_samples,
    latent_dim=10,
    hidden_dim=128,
    batch_size=64,
    epochs=100,
    lr=1e-4,
    betas=(0.0, 0.9),
    lambda_gp=10.0,
    n_critic=5,
    device=None,
    random_state=None,
):
    """Train a simple WGAN-GP on X_minority and generate n_samples synthetic examples.

    Returns numpy array of generated samples with shape (n_samples, feature_dim).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if random_state is not None:
        torch.manual_seed(int(random_state))
        np.random.seed(int(random_state))

    X = np.asarray(X_minority, dtype=np.float32)
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    feature_dim = X.shape[1]
    gen = Generator(latent_dim, feature_dim, hidden_dim).to(device)
    critic = Critic(feature_dim, hidden_dim).to(device)

    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=betas)
    opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=betas)

    iterations = 0
    for epoch in range(epochs):
        for batch in loader:
            real = batch[0].to(device)

            # train critic n_critic times
            for _ in range(n_critic):
                z = torch.randn(real.size(0), latent_dim, device=device)
                fake = gen(z).detach()
                critic_real = critic(real)
                critic_fake = critic(fake)
                gp = _gradient_penalty(critic, real, fake, device=device)
                loss_critic = -(critic_real.mean() - critic_fake.mean()) + lambda_gp * gp

                opt_critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

            # train generator
            z = torch.randn(real.size(0), latent_dim, device=device)
            fake = gen(z)
            loss_gen = -critic(fake).mean()
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            iterations += 1

    # generation
    gen.eval()
    samples = []
    with torch.no_grad():
        n_batches = int(math.ceil(n_samples / batch_size))
        for _ in range(n_batches):
            z = torch.randn(batch_size, latent_dim, device=device)
            out = gen(z)
            out = out.cpu().numpy()
            samples.append(out)
    samples = np.vstack(samples)[:n_samples]
    return samples
