# Mask preview: training map 50052748

Four snapshots from one accumulated lidar trajectory, seed 42. Each PNG is
429 pixels wide and 218 pixels high. No generated map completions are included.

- `sample_00_mask.png` through `sample_03_mask.png`: white = unknown,
  black = observed. These are binary masks with pixel values 255 and 0.
- `sample_00_observation.png` through `sample_03_observation.png`: black = wall,
  white = free, gray (128) = unknown. These are visualization exports; use the
  binary masks for training rather than inferring unknowns from PNG intensity.
- `ground_truth.png`: complete target, black = wall and white = free.
- `metadata.json`: sensor settings and snapshot statistics.

| Sample | Accumulated scans | Observed valid-space coverage |
| --- | ---: | ---: |
| 00 | 1 | 17.55% |
| 01 | 31 | 38.09% |
| 02 | 91 | 38.50% |
| 03 | 180 | 39.21% |

The preview is not a finished training dataset. Its routing mask intersects
free space with the valid-space annotation and then keeps only the largest
connected region. That intersection has 26 components on the processed map;
even the raw-resolution intersection has 28, so downsampling alone does not
explain the disconnection. The route repeatedly samples random destinations
within the retained region, giving little new visibility after the early scans.
Navigation traversability and evaluation valid-space annotations need separate
treatment before full dataset generation. Conservative downsampling and robot
clearance can also close narrow passages and require explicit checks.

Ground-truth shortest paths were used solely to create this preview. Visible
labels are copied from the ground truth at lidar-visible cells to eliminate
raycasting boundary label artifacts, making this an idealized noiseless example.
All snapshots share the same ground truth, and later snapshots are highly
correlated. They demonstrate masking and alignment, not broad training coverage
or calibrated aleatoric uncertainty.
