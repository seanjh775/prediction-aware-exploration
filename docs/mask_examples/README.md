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
- `preview.png`: side-by-side comparison with trajectories and coverage.

| Sample | Accumulated scans | Observed valid-space coverage |
| --- | ---: | ---: |
| 00 | 1 | 7.25% |
| 01 | 54 | 36.69% |
| 02 | 160 | 81.46% |
| 03 | 319 | 92.97% |

The preview is not a finished training dataset. Navigation now comes from the
occupancy map and an explicit circular footprint; valid-space annotations only
select the interior component, sample endpoints, and measure coverage. Routes
can traverse free connections outside the annotation. Border-connected free
space is excluded so the padded exterior cannot become the selected component.

The synthetic footprint radius defaults to 0.1 m (one processed pixel at the
configured 10 pixels/m), not a measured robot specification. Configure it with
`--robot-radius-m`; clearance requires distance to occupied cell centers strictly
greater than the radius. This raster approximation and conservative map
downsampling need verification against the intended robot. The selection rejects
maps with no non-border-connected annotated component rather than silently
routing through the exterior.

The original preview clipped routes to the annotation, splitting navigable
space into 26 regions, and stalled near 39% coverage. The corrected preview
reaches about 93% using the same map and seed, but a different start/trajectory
because the candidate navigation region changed. This is not a controlled
equal-scan-count policy comparison.

Ground-truth shortest paths were used solely to create this preview. Visible
labels are copied from the ground truth at lidar-visible cells to eliminate
raycasting boundary label artifacts, making this an idealized noiseless example.
All snapshots share the same ground truth. The route is still a random sequence
of oracle shortest paths, not an observation-only exploration policy. Multiple
starts and coverage-aware snapshot selection remain work for the full dataset.
These examples demonstrate masking and alignment, not broad training coverage
or calibrated aleatoric uncertainty.
