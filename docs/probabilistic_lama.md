# Single-pass probabilistic Big LaMa

This baseline predicts one occupancy logit per cell in a single deterministic
evaluation pass. It uses the pretrained Big LaMa FFC backbone, with latent noise
disabled. No WGAN critic, adversarial loss, feature-matching loss, perceptual loss,
class weights, or focal loss participates in this training path. Existing GAN and
exploration code is preserved separately.

## Encoding and pretrained adaptation

Pilot targets use 0 = free and 1 = occupied. The map-specific RGB checkpoint also uses
occupied = 1 and free = 0. The adapter repeats known occupancy into RGB channels,
removes unknown content, and appends the unknown mask. The PNG review gallery
inverts this convention for display only; it is not the model input encoding.
Thus hidden target labels cannot enter the generator conditioning input.

All backbone parameters load strictly from the Big LaMa checkpoint. The last
three-channel convolution is replaced by a one-channel convolution without a
sigmoid. Its weights and bias start from the mean of the RGB occupancy output
weights and bias, retaining their original sign. This
is an initialization, not a guarantee of calibrated probabilities. There is no
separate learned uncertainty head and no ensemble or latent-sample comparison.

## Objective and data

The primary loss is unweighted `binary_cross_entropy_with_logits`, summed over
unknown real-image cells and divided by their count. Known cells and computational
padding contribute zero loss. The eligibility mask is **not** the valid-space
annotation: real connecting corridors and exterior image cells are not silently
removed. All-zero eligibility is rejected.

`OccupancyDataset` reads the fixed pilot manifest and map assignment. Training
uses seeded, epoch-dependent, aligned crops without rescaling; each crop contains
an unknown anchor. This deliberately samples crops with unknown content, so any
calibration claim must be assessed on the held-out full-image distribution.
The collator pads heterogeneous shapes to multiples of 16, tracks actual image
support, and uses free/known padding (black in pretrained RGB space).

By default validation uses complete processed maps, one per batch, and no random
augmentation. Development uses only train and validation splits; this runner
does not load test examples. An optional fixed validation crop and example limit
are provided for smoke checks, not final evaluation.

## Run

From the repository root, with the existing project Python environment:

```bash
PYTHONPATH=.:lama python scripts/train_probabilistic_lama.py \
  --output work/probabilistic_run_01 \
  --steps 1000 --crop-size 256 --batch-size 1 --lr 0.00001
```

CUDA is selected when available; otherwise CPU is used. Big LaMa is substantial,
so full fine-tuning and full-map validation can be slow on CPU. Crop size is in
processed map pixels and does not change the physical pixel scale.

The output directory must be new. `config.json` records architecture and settings;
`metrics.jsonl` logs steps and validation; `last.pt` and `best.pt` save model and
optimizer state. Best means lowest validation NLL. Before training, the runner logs validation at step zero and includes that baseline
in best-checkpoint selection. `--train-maps` limits training to explicitly named
training-map IDs. `--validation-samples-per-map` selects fixed coverage-spaced
examples from each held-out validation map for short runs. The script does not yet resume
an interrupted optimizer run. No claim of exact cross-device reproducibility is
made. Full training runs are not started automatically by implementation checks.

For a short CPU mechanics check:

```bash
PYTHONPATH=.:lama python scripts/train_probabilistic_lama.py \
  --output work/probabilistic_smoke_new --device cpu \
  --steps 1 --crop-size 64 --validation-crop-size 64 \
  --validation-limit 1 --validate-every 1
```

Small crops may remove essentially all observation context. Their initial
confidence and metrics are not representative of full-map model performance.

## Inference and evaluation

```python
from models.probabilistic_lama import load_occupancy_checkpoint

model = load_occupancy_checkpoint('work/probabilistic_run_01/best.pt', device='cpu')
result = model.predict(observation, unknown_mask)  # matching float Bx1xHxW tensors
probability = result['completed_probability']
variance = result['variance']
entropy = result['entropy']
```

Logits are converted to occupancy probabilities using sigmoid. The raw
`occupancy_probability` contains network predictions everywhere;
`completed_probability` preserves known observations exactly. Variance `p(1-p)`
and binary entropy (natural-log units) are zeroed at known cells. An inference
consumer batching padded maps should also discard/mask computational padding.
These uncertainty outputs estimate learned conditional occupancy ambiguity;
they do not automatically separate aleatoric ambiguity from model error.

Validation logs unknown non-padding-cell NLL, Brier score, threshold-0.5 accuracy,
occupied-cell IoU, and a ten-bin reliability table plus ECE. IoU is null if the
occupied union is empty. Metrics are reported globally, per source map, and by
observed-image coverage quartile; coverage excludes padding and is different
from the collection's valid-space coverage. Pixel-weighted scores are descriptive,
not independent statistical samples, because pixels and trajectories correlate.

Run the portable tests with:

```bash
PYTHONPATH=.:lama python -m unittest discover -s tests -v
```

Encoding correction: checkpoint format version 2 uses direct occupancy inputs
and a positive mean output head. Earlier smoke checkpoints used an inverted
encoding and must not be resumed; the inference loader rejects those checkpoints.
