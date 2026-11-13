# Update-10 versus update-20 loss diagnosis

The fixed policy remains 10 training maps, 2 validation maps, and 2 reserved test maps. This investigation evaluated the same six validation crops as the short run. No gradient updates were performed, and no test data was loaded.

## Where the increase occurs

| Validation sample | NLL at 10 | NLL at 20 | Share of net loss increase |
| --- | ---: | ---: | ---: |
| 50015847_t02_s05 | 1.47324 | 15.58972 | 92.55% |
| 50015847_t00_s02 | 2.83328 | 3.26444 | 3.90% |
| 50015847_t03_s00 | 2.70797 | 2.89128 | 1.88% |
| 50015848_t02_s00 | 2.15768 | 2.26109 | 0.90% |
| 50015848_t00_s04 | 0.41701 | 0.51374 | 0.52% |
| 50015848_t01_s01 | 0.72686 | 0.76504 | 0.25% |

The dominant crop is `50015847_t02_s05`, at processed-map origin (0, 0). Its largest worsened-error region contains 241 cells within rows [0, 13), columns [184, 256), near the top-right of this 256x256 crop. This region accounts for 88.5% of the net increase across all 249,694 evaluated cells. The worst 1% of cells account for 99.2% of the net increase. Coordinates are zero-based processed-map pixels, with upper bounds excluded.

Most deterioration is wall targets being assigned increasingly confident free-space predictions. Occupied-target mean NLL rises from 10.97 to 26.76; free-target mean NLL improves from 0.423 to 0.331. There are 1,948 newly incorrect cells and 1,664 corrected cells, a net increase of just 284 incorrect classifications. Previously incorrect cells becoming more confident dominate the loss increase. Brier is bounded per cell, whereas Bernoulli NLL is unbounded.

At crop cell (row 1, column 251), the target is occupied. Its occupancy logit changes from -16.10 to -32159.50, and cell NLL from 16.10 to 32159.50. The corresponding probability is effectively zero at update 20; this is an extreme confidence/activation failure.

## Evaluation-only normalization-buffer swap

Weights include convolution parameters and learned normalization scale/bias. Statistics means only BatchNorm running_mean, running_var, and num_batches_tracked. Every case uses eval mode and identical input arrays.

| Learned weights | Running statistics | Validation NLL |
| --- | --- | ---: |
| 10 | 10 | 1.909645 |
| 20 | 20 | 4.057059 |
| 20 | 10 | 1.894416 |
| 10 | 20 | 4.049804 |

These results isolate changed running statistics as the dominant source of the regression on these six crops. They do not establish performance on full maps or guarantee that frozen-statistics training will improve every case. The hybrid models are diagnostic combinations, not newly trained replacements.

The instantiated network contains 152 BatchNorm layers, each with momentum 0.1 and epsilon 1e-5. Every BatchNorm batch counter increases by 10 between the checkpoints. The running-mean update is 90% old reference plus 10% current batch reference. Ten updates retain about 34.9% of the earlier reference.

## Code locations

- `lama/saicinpainting/training/modules/ffc.py:306`: default BatchNorm for the generator; line 308 for upsampling.
- `lama/saicinpainting/training/modules/ffc.py:233`: local/global FFC normalization; lines 243-254 construct and apply it.
- `lama/saicinpainting/training/modules/ffc.py:60`: Fourier-unit BatchNorm; line 101 applies it.
- `lama/saicinpainting/training/modules/ffc.py:131`: spectral-transform BatchNorm.
- `scripts/train_probabilistic_lama.py:45`: validation switches to evaluation mode.
- `scripts/train_probabilistic_lama.py:128`: each training batch switches the model back to training mode.

Next controlled experiment: repeat the original short run with identical seed, training order, crops, learning rate and validation set, while setting only BatchNorm modules to eval after each model.train() call. Keep their learned scale and bias trainable. That experiment has not yet been run.

Reproduce the diagnosis with `PYTHONPATH=.:lama python scripts/diagnose_occupancy_loss.py --output work/loss_diagnosis_new`. Full diagnostic arrays stay local in `work/loss_diagnosis/`; no new map images or model checkpoints are published. [Aggregate diagnostic metrics](normalization_diagnosis_metrics.json).
