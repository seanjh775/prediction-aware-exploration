# Latent-conditioned LaMa

The FFC generator accepts an optional Gaussian latent vector. A trainable linear
projection turns the vector into a coarse spatial field, upsamples it, and adds
a bounded perturbation to the masked image channels inside unknown regions.
The mask channel is unchanged. The existing training module composites the final
completion with the original observation, preserving known cells exactly.

This is an input-conditioning baseline for subsequent conditional WGAN training.
It demonstrates sensitivity to latent input, not learned layout diversity or
calibrated aleatoric uncertainty. No diversity loss or WGAN loss is added here.

## Configuration

Add these fields to the existing `generator` section of a model configuration:

```yaml
latent_dim: 32
latent_grid_size: 8
latent_strength: 0.1
```

The defaults disable conditioning (`latent_dim: 0`). Enabled conditioning requires
an FFC generator with image channels followed by one mask channel, and training
settings `concat_mask: true` and `add_noise_kwargs: null`. Do not combine it with
LaMa's older multiscale-noise option.

Existing generator parameter shapes and names are preserved. Loading a deterministic
checkpoint into the latent-enabled generator with `strict=False` leaves only the
new `latent_projection.weight` uninitialized from the checkpoint. It is randomly
initialized, deliberately nonzero so the latent path has an immediate effect.
Save the updated configuration alongside fine-tuned checkpoints; subsequent loads
with matching configurations restore the projection as well.

The projection belongs to the generator and is included in the existing generator
optimizer. Fine-tune the generator; the inference helper freezes its returned model
and is not a training entry point. Training objectives that encourage meaningful
latent usage remain a separate step.

## Inference

The existing loader can enable conditioning on an old checkpoint:

```python
model = load_lama_model(
    model_path, device='cpu',
    latent_kwargs={'latent_dim': 32, 'latent_grid_size': 8, 'latent_strength': 0.1},
)
z = torch.randn(batch['image'].shape[0], 32,
                device=batch['image'].device, dtype=batch['image'].dtype)
with torch.no_grad():
    output = model(dict(batch, latent=z))['inpainted']
```

Omitting `latent` samples a new vector each forward pass, including in evaluation
mode. Supply the same vector to replay a sample in evaluation mode. A zero vector
recovers the original deterministic generator output because the projection has
no bias. Replaying across processes also requires restoring the same projection
weights, or the same initialization seed when adapting an old checkpoint.

`model_path` must contain `config.yaml` and `models/best.ckpt`, as before.

## Verification

From the repository root, using the project's Python environment:

```bash
PYTHONPATH=lama python -m unittest discover -s tests -v
```

Checks cover deterministic-checkpoint compatibility, latent replay and sensitivity,
zero-latent equivalence, gradients through the conditioning path, exact preservation
of known cells after compositing, and latent-enabled state restoration.
