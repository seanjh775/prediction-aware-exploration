# GAN reliability branch handoff

Goal: estimate conditional ambiguity in unseen occupancy maps using multiple
samples from one latent-conditioned generator, not a model ensemble or the
critic score itself. Calibration and meaningful structural diversity remain
unproven; noise sensitivity alone is not aleatoric uncertainty.

Implemented:

- Optional latent conditioning for LaMa's FFC generator, with existing generator
  weight shapes preserved. See `latent_lama.md`.
- Standalone conditional scalar critic and WGAN-GP objectives, including a
  gradient path to LaMa's latent projection. See `wgan_critic.md`.
- Fixed map split in `configs/critic_map_split.json`: 10 training, 2 validation,
  2 test. Building groupings are provisional, based on map identifiers.
- A one-map masked observation preview, not a full training dataset.

Run the tests from the repository root in the project's Python environment:

```bash
PYTHONPATH=lama python -m unittest discover -s tests -v
```

Regenerate the preview (requires the existing simulation dependencies and maps):

```bash
PYTHONPATH=.:scripts python scripts/generate_mask_preview.py
```

The script writes `work/masked_map_preview/preview.png`, metadata, and four NPZ
samples for training map `50052748`. It uses the existing lidar simulator and
oracle shortest paths for this preview, not an observation-only exploration
policy. Known labels are taken from ground truth at lidar-visible cells to
avoid raycasting boundary label artifacts. The trajectory stays within one
connected area; later snapshots are similar and coverage stalls near 39%.
Improve trajectory diversity, valid-space connectivity handling, and robot
footprint/resolution decisions before scaling up dataset generation. The user
has not yet visually approved the preview because chat image rendering failed.

Next work:

1. Review the preview in a local image viewer or VS Code.
2. Build reproducible multi-map observation generation and a paired dataset
   loader. All observations and crops inherit the source map split. The saved
   run on `50015848` is validation-only.
3. Add a checkpointed/logged critic-training runner and run a short frozen-LaMa
   smoke test. No WGAN training run has started.
4. Connect joint generator/critic updates and an explicit latent-diversity
   objective; the current LaMa adversarial training loop is still unchanged.
5. Evaluate sample plausibility, diversity, and occupancy uncertainty calibration
   on held-out maps before using uncertainty for planning.

Checkpoints and generated data are not tracked by Git. The cloud environment
uses `/workspace/setup-tools/pipe` (Python 3.6) with extra simulation dependencies.
For another machine, start from the repository environment instructions and
install/build `range_libc`. The extracted pretrained model `config.yaml` files
were missing during verification; the Big LaMa config was read from the local
weights archive for the integration check. Restore matching configs alongside
checkpoints before using the normal inference loader. The original LaMa
training dataset referenced under `/ocean/projects/...` is not mounted here.
