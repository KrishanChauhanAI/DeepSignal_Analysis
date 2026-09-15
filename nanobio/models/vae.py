"""
Variational Autoencoder for noise characterization.

Stub — enabled via config.models.vae_enabled. Full implementation left
as an extension point for the noise-analysis module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignalVAE(nn.Module):
    """Simple 1-D convolutional VAE for signal windows."""

    def __init__(self, input_length: int = 1024, latent_dim: int = 32):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_lv = nn.Linear(128, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 128 * (input_length // 8))
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 7, stride=2, padding=3, output_padding=1),
        )

    def reparameterize(self, mu, lv):
        std = torch.exp(0.5 * lv)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, lv = self.fc_mu(h), self.fc_lv(h)
        z = self.reparameterize(mu, lv)
        d = self.fc_dec(z).view(-1, 128, self.input_length // 8)
        return self.decoder(d), mu, lv

    @staticmethod
    def loss(recon_x, x, mu, lv, beta: float = 1.0):
        recon = F.mse_loss(recon_x, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
        return recon + beta * kld