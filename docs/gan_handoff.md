# GAN reliability branch handoff

Current goal: estimate conditional occupancy ambiguity in one pass, without an
ensemble or comparing generated map samples. The approved baseline fine-tunes
pretrained Big LaMa with one occupancy-logit output and masked unweighted BCE.
Probability variance/entropy supplies uncertainty; calibration remains unproven.
See `probabilistic_lama.md`. The earlier latent WGAN code is retained but is not
used by this baseline.

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

1. A 20-update experiment on three training maps is complete; see
   `training_runs/probabilistic_short_run.md`. The best checkpoint was step 10;
   validation NLL worsened later. Investigate stability and expand held-out
   validation before a longer run or final test evaluation. The input/head
   encoding was corrected to occupied=1/free=0; use checkpoint encoding version 2.
2. Review `docs/pilot_review/README.md` for the pilot sample gallery and coverage
   results. Generation and validation are documented in `docs/masked_pilot.md`.
   The full NPZ dataset stays under `data/kth_masked_pilot_v1/` in the cloud,
   excluded from Git. The paired loader and probabilistic runner now exist,
   preserving the source map split and aligned arrays. The old saved run on `50015848` is
   validation-only. One cropped test map needs an explicit image-extent virtual
   boundary for routing; inspect its route statistics before broader use.
3. Evaluate single-pass probabilities and uncertainty before connecting them to
   planning. A critic is not required for the approved probabilistic baseline.
   WGAN training and latent-diversity objectives are deferred.

Checkpoints and generated data are not tracked by Git. The cloud environment
uses `/workspace/setup-tools/pipe` (Python 3.6) with extra simulation dependencies.
For another machine, start from the repository environment instructions and
install/build `range_libc`. The extracted pretrained model `config.yaml` files
were missing during verification; the Big LaMa config was read from the local
weights archive for the integration check. Restore matching configs alongside
checkpoints before using the normal inference loader. The original LaMa
training dataset referenced under `/ocean/projects/...` is not mounted here.
