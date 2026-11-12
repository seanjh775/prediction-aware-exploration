# Conditional WGAN-GP critic

`ConditionalMapCritic` accepts a candidate map, partial observation, and binary
unknown mask (1 = unknown). It outputs one unrestricted scalar per example;
higher scores are encouraged for real maps. Scores are neither probabilities
nor aleatoric uncertainty estimates. Uncertainty will come from calibrated
diversity among latent-conditioned generator samples.

The critic concatenates the completed candidate, known observation, and mask.
It replaces candidate values in known regions with the observation and zeros
unknown placeholders in the observation condition. This prevents discrimination
based on differences in known cells or placeholder conventions. Use the same
condition for each real/generated pair. Inputs must use matching BCHW shapes,
device, dtype, and normalization (the current LaMa maps use [0, 1]). The default
critic accepts RGB maps, including repeated grayscale channels used by LaMa.

Strided convolutions and global pooling support variable map sizes of at least
16 pixels per side in the default configuration. There is no sigmoid or batch
normalization. During a critic step the minimized objective is:

`mean(fake_score) - mean(real_score) + 10 * gradient_penalty`

The penalty interpolates real/generated candidates under the same condition and
targets a gradient norm of 1. Known candidate cells have zero gradients because
of compositing, so the norm measures sensitivity to the unknown completion.
Examples without unknown cells are excluded from training.

## Minimal update sequence

Settings are in `configs/wgan_critic.yaml`. This is a standalone configuration,
not a replacement for LaMa's Hydra settings or its existing discriminator.

```python
import torch
import yaml
from saicinpainting.training.modules.conditional_wgan import ConditionalMapCritic
from saicinpainting.training.losses.wasserstein import (
    critic_loss, generator_adversarial_loss,
)

with open('configs/wgan_critic.yaml') as stream:
    config = yaml.safe_load(stream)
critic = ConditionalMapCritic(**config['critic']).to(device)
critic_optimizer = torch.optim.Adam(critic.parameters(), **config['optimizer'])

# real, observation, mask: paired tensors from the TRAIN split only.
# generator: latent-enabled LaMa training module (not a frozen inference model).
for parameter in critic.parameters():
    parameter.requires_grad_(True)
critic_optimizer.zero_grad(set_to_none=True)
with torch.no_grad():
    fake = generator({'image': observation, 'mask': mask})['inpainted']
loss, metrics = critic_loss(
    critic, real, fake, observation, mask,
    gp_weight=config['training']['gradient_penalty_weight'],
)
loss.backward()
critic_optimizer.step()

# Once joint training begins, take about five critic steps per generator step.
# Use fresh batches/latent samples for the critic updates.
for parameter in critic.parameters():
    parameter.requires_grad_(False)
generator_optimizer.zero_grad(set_to_none=True)
fake = generator({'image': observation, 'mask': mask})['inpainted']
adversarial = generator_adversarial_loss(critic, fake, observation, mask)
# This is only the adversarial component. A full generator objective still
# needs appropriate map fidelity and latent diversity terms.
adversarial.backward()
generator_optimizer.step()
```

If freezing the generator for critic-only warmup, also put it in evaluation mode
so its batch-normalization running statistics do not change. Restore training
mode for joint generator training. WGAN-GP requires autograd even during a
critic-only phase; do not wrap the critic loss in `no_grad`.

Log real/fake scores, their gap, gradient penalty, and held-out map quality.
The critic loss alone is not a completion-quality or uncertainty metric.

## Current scope

This adds the critic, losses, configuration, and verification. It does not modify
the legacy LaMa adversarial training loop, prepare observation datasets, or launch
a training job. The next step is a paired dataset loader and training runner using
`configs/critic_map_split.json`; every crop or observation inherits its map split.
The existing `50015848` exploration run is validation data, not training data.

Run checks from the repository root:

```bash
PYTHONPATH=lama python -m unittest discover -s tests -v
```
