"""A scalar critic conditioned on an observed map and its unknown mask."""
import torch
from torch import nn


class ConditionalMapCritic(nn.Module):
    def __init__(self, image_channels=3, base_channels=64, num_layers=4,
                 max_channels=512):
        super().__init__()
        if min(image_channels, base_channels, num_layers, max_channels) < 1:
            raise ValueError('Critic dimensions must be positive')
        self.image_channels = image_channels
        self.min_size = 2 ** num_layers
        layers = []
        channels = 2 * image_channels + 1
        for index in range(num_layers):
            out_channels = min(base_channels * 2 ** index, max_channels)
            layers.extend([nn.Conv2d(channels, out_channels, 4, stride=2, padding=1),
                           nn.LeakyReLU(0.2)])
            channels = out_channels
        # No batch normalization: examples must remain independent for WGAN-GP.
        # No sigmoid: Wasserstein scores are unrestricted, not probabilities.
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.score = nn.Linear(channels, 1)

    def forward(self, candidate, observation, unknown_mask):
        if candidate.ndim != 4 or candidate.shape != observation.shape:
            raise ValueError('candidate and observation must have matching BCHW shapes')
        if candidate.shape[1] != self.image_channels:
            raise ValueError('Unexpected image channel count')
        if unknown_mask.shape != (candidate.shape[0], 1, *candidate.shape[-2:]):
            raise ValueError('unknown_mask must have shape (B, 1, H, W)')
        if min(candidate.shape[-2:]) < self.min_size:
            raise ValueError('Map is too small for the configured number of layers')
        if any(t.device != candidate.device or t.dtype != candidate.dtype
               for t in (observation, unknown_mask)):
            raise ValueError('All inputs must share device and floating dtype')
        if not candidate.is_floating_point():
            raise ValueError('Maps must be floating point')
        if not torch.all((unknown_mask == 0) | (unknown_mask == 1)):
            raise ValueError('unknown_mask must be binary: 1 means unknown')
        known = observation * (1 - unknown_mask)
        completed = candidate * unknown_mask + known
        conditioned = torch.cat([completed, known, unknown_mask], dim=1)
        return self.score(self.pool(self.features(conditioned)).flatten(1)).squeeze(1)
