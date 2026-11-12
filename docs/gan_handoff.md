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
- A one-map masked observation preview, plus multi-map pilot generation,
  split-safe validation, and a reproducible random 20% PNG review export.

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
avoid raycasting boundary label artifacts. Routing now uses occupancy and an
explicit circular footprint, with valid-space annotations used for endpoints
and coverage rather than clipping paths. The corrected preview reaches 93%
coverage instead of the original 39%; exported PNGs are in `docs/mask_examples`.
The default footprint radius is a synthetic 0.1 m assumption, configurable using
`--robot-radius-m`. Verify it against the intended robot and processed map scale.
Improve trajectory diversity and coverage-aware snapshot selection before
scaling up dataset generation. The revised preview still needs user review.

Next work:

1. Review the preview in a local image viewer or VS Code.
2. Review `docs/pilot_review/README.md` for the pilot sample gallery and coverage
   results. Generation and validation are documented in `docs/masked_pilot.md`.
   The full NPZ dataset stays under `data/kth_masked_pilot_v1/` in the cloud,
   excluded from Git. Build a paired training dataset loader next, preserving
   the source map split and aligned arrays. The old saved run on `50015848` is
   validation-only. One cropped test map needs an explicit image-extent virtual
   boundary for routing; inspect its route statistics before broader use.
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
