# Masked pilot: random sample review

420 examples generated across 14 maps; 25 examples published from the seeded random review. The full 20% review (84 examples) remains in the cloud workspace. This gallery is a random subset of that review, sampled without replacement using subset seed `20261113` (parent selection seed `20261112`).

The review is drawn from saved examples, not maps. All four trajectories per map use a 0.1 m synthetic circular footprint. Target coverage levels can be skipped when one scan crosses several levels or a trajectory exhausts its budget.

| Split | Generated | In random review |
| --- | ---: | ---: |
| train | 328 | 21 |
| validation | 36 | 1 |
| test | 56 | 3 |

Each contact sheet shows ground truth, partial observation, and unknown mask. Observation: black wall, white free, gray hidden. Mask: white hidden, black known. Test examples are included for dataset-integrity inspection only, not model tuning.

## Contact sheets

- [Page 1 — selected examples 1–6](contact_sheet_01.png)
- [Page 2 — selected examples 7–12](contact_sheet_02.png)
- [Page 3 — selected examples 13–18](contact_sheet_03.png)
- [Page 4 — selected examples 19–24](contact_sheet_04.png)
- [Page 5 — selected examples 25–25](contact_sheet_05.png)

Navigation exception: `50037764_PLAN1` uses the original image extent as a virtual boundary because its cropped floor opens into simulation padding. Other maps use a closed occupancy component. See the dataset documentation and trajectory details for this limitation.

## Map collection results

| Map | Split | Saved examples | Final trajectory coverage |
| --- | --- | ---: | --- |
| 50010535_PLAN1 | train | 27 | 79.6%, 67.0%, 78.0%, 77.2% |
| 50010535_PLAN2 | train | 23 | 73.8%, 50.7%, 77.4%, 56.0% |
| 50010536_PLAN3 | train | 21 | 63.4%, 54.5%, 45.9%, 60.3% |
| 50052748 | train | 38 | 95.1%, 94.1%, 95.2%, 95.1% |
| 50052749 | train | 31 | 90.2%, 88.0%, 90.1%, 86.7% |
| 50052750 | train | 36 | 91.7%, 92.9%, 93.2%, 93.5% |
| 50052751 | train | 38 | 95.2%, 95.0%, 95.0%, 94.2% |
| 50052752 | train | 40 | 95.0%, 95.1%, 95.3%, 95.1% |
| 50052753 | train | 39 | 95.0%, 95.1%, 94.0%, 95.1% |
| 50052754 | train | 35 | 94.8%, 94.8%, 94.6%, 94.3% |
| 50015847 | validation | 21 | 50.5%, 50.9%, 64.7%, 56.4% |
| 50015848 | validation | 15 | 52.2%, 37.7%, 38.9%, 42.3% |
| 50037764_PLAN1 | test | 27 | 80.9%, 57.7%, 78.2%, 79.4% |
| 50037765_PLAN3 | test | 29 | 85.7%, 83.2%, 73.7%, 68.3% |

## Individual PNGs

| Sample | Split | Coverage | Target | Observation | Mask |
| --- | --- | ---: | --- | --- | --- |
| 50010535_PLAN1_t00_s00 | train | 10.3% | [PNG](examples/50010535_PLAN1_t00_s00_ground_truth.png) | [PNG](examples/50010535_PLAN1_t00_s00_observation.png) | [PNG](examples/50010535_PLAN1_t00_s00_mask.png) |
| 50010535_PLAN2_t00_s00 | train | 10.1% | [PNG](examples/50010535_PLAN2_t00_s00_ground_truth.png) | [PNG](examples/50010535_PLAN2_t00_s00_observation.png) | [PNG](examples/50010535_PLAN2_t00_s00_mask.png) |
| 50010535_PLAN2_t00_s03 | train | 40.1% | [PNG](examples/50010535_PLAN2_t00_s03_ground_truth.png) | [PNG](examples/50010535_PLAN2_t00_s03_observation.png) | [PNG](examples/50010535_PLAN2_t00_s03_mask.png) |
| 50010536_PLAN3_t02_s02 | train | 30.3% | [PNG](examples/50010536_PLAN3_t02_s02_ground_truth.png) | [PNG](examples/50010536_PLAN3_t02_s02_observation.png) | [PNG](examples/50010536_PLAN3_t02_s02_mask.png) |
| 50010536_PLAN3_t03_s05 | train | 60.3% | [PNG](examples/50010536_PLAN3_t03_s05_ground_truth.png) | [PNG](examples/50010536_PLAN3_t03_s05_observation.png) | [PNG](examples/50010536_PLAN3_t03_s05_mask.png) |
| 50052748_t03_s06 | train | 70.2% | [PNG](examples/50052748_t03_s06_ground_truth.png) | [PNG](examples/50052748_t03_s06_observation.png) | [PNG](examples/50052748_t03_s06_mask.png) |
| 50052749_t01_s06 | train | 83.2% | [PNG](examples/50052749_t01_s06_ground_truth.png) | [PNG](examples/50052749_t01_s06_observation.png) | [PNG](examples/50052749_t01_s06_mask.png) |
| 50052750_t02_s01 | train | 21.1% | [PNG](examples/50052750_t02_s01_ground_truth.png) | [PNG](examples/50052750_t02_s01_observation.png) | [PNG](examples/50052750_t02_s01_mask.png) |
| 50052750_t02_s08 | train | 90.2% | [PNG](examples/50052750_t02_s08_ground_truth.png) | [PNG](examples/50052750_t02_s08_observation.png) | [PNG](examples/50052750_t02_s08_mask.png) |
| 50052751_t00_s01 | train | 20.7% | [PNG](examples/50052751_t00_s01_ground_truth.png) | [PNG](examples/50052751_t00_s01_observation.png) | [PNG](examples/50052751_t00_s01_mask.png) |
| 50052751_t00_s06 | train | 70.8% | [PNG](examples/50052751_t00_s06_ground_truth.png) | [PNG](examples/50052751_t00_s06_observation.png) | [PNG](examples/50052751_t00_s06_mask.png) |
| 50052751_t01_s07 | train | 80.7% | [PNG](examples/50052751_t01_s07_ground_truth.png) | [PNG](examples/50052751_t01_s07_observation.png) | [PNG](examples/50052751_t01_s07_mask.png) |
| 50052751_t03_s05 | train | 77.6% | [PNG](examples/50052751_t03_s05_ground_truth.png) | [PNG](examples/50052751_t03_s05_observation.png) | [PNG](examples/50052751_t03_s05_mask.png) |
| 50052752_t01_s05 | train | 62.8% | [PNG](examples/50052752_t01_s05_ground_truth.png) | [PNG](examples/50052752_t01_s05_observation.png) | [PNG](examples/50052752_t01_s05_mask.png) |
| 50052752_t01_s06 | train | 72.8% | [PNG](examples/50052752_t01_s06_ground_truth.png) | [PNG](examples/50052752_t01_s06_observation.png) | [PNG](examples/50052752_t01_s06_mask.png) |
| 50052753_t01_s02 | train | 30.0% | [PNG](examples/50052753_t01_s02_ground_truth.png) | [PNG](examples/50052753_t01_s02_observation.png) | [PNG](examples/50052753_t01_s02_mask.png) |
| 50052753_t01_s08 | train | 90.2% | [PNG](examples/50052753_t01_s08_ground_truth.png) | [PNG](examples/50052753_t01_s08_observation.png) | [PNG](examples/50052753_t01_s08_mask.png) |
| 50052753_t01_s09 | train | 95.1% | [PNG](examples/50052753_t01_s09_ground_truth.png) | [PNG](examples/50052753_t01_s09_observation.png) | [PNG](examples/50052753_t01_s09_mask.png) |
| 50052754_t01_s01 | train | 30.6% | [PNG](examples/50052754_t01_s01_ground_truth.png) | [PNG](examples/50052754_t01_s01_observation.png) | [PNG](examples/50052754_t01_s01_mask.png) |
| 50052754_t01_s03 | train | 52.6% | [PNG](examples/50052754_t01_s03_ground_truth.png) | [PNG](examples/50052754_t01_s03_observation.png) | [PNG](examples/50052754_t01_s03_mask.png) |
| 50052754_t02_s00 | train | 10.2% | [PNG](examples/50052754_t02_s00_ground_truth.png) | [PNG](examples/50052754_t02_s00_observation.png) | [PNG](examples/50052754_t02_s00_mask.png) |
| 50015848_t00_s01 | validation | 20.2% | [PNG](examples/50015848_t00_s01_ground_truth.png) | [PNG](examples/50015848_t00_s01_observation.png) | [PNG](examples/50015848_t00_s01_mask.png) |
| 50037765_PLAN3_t01_s03 | test | 40.2% | [PNG](examples/50037765_PLAN3_t01_s03_ground_truth.png) | [PNG](examples/50037765_PLAN3_t01_s03_observation.png) | [PNG](examples/50037765_PLAN3_t01_s03_mask.png) |
| 50037765_PLAN3_t02_s02 | test | 30.4% | [PNG](examples/50037765_PLAN3_t02_s02_ground_truth.png) | [PNG](examples/50037765_PLAN3_t02_s02_observation.png) | [PNG](examples/50037765_PLAN3_t02_s02_mask.png) |
| 50037765_PLAN3_t02_s05 | test | 60.1% | [PNG](examples/50037765_PLAN3_t02_s05_ground_truth.png) | [PNG](examples/50037765_PLAN3_t02_s05_observation.png) | [PNG](examples/50037765_PLAN3_t02_s05_mask.png) |

[Selection manifest](selection.json) · [Validation report](validation_report.json) · [Trajectory details](map_summary.json)

Complete NPZ data is stored in the cloud workspace under `data/kth_masked_pilot_v1/` and excluded from Git. This gallery contains the selected PNGs only. See [dataset documentation](../masked_pilot.md) for generation and validation commands.
