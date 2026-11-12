# KTH masked-observation pilot

This dataset prepares observations for conditional map completion and critic
training. It contains no generated completions or trained critic outputs.

From the repository root, in the existing project Python environment:

```bash
PYTHONPATH=.:scripts python scripts/generate_masked_pilot.py --workers 2
PYTHONPATH=.:scripts python scripts/validate_masked_pilot.py data/kth_masked_pilot_v1
PYTHONPATH=.:scripts python scripts/export_pilot_review.py data/kth_masked_pilot_v1 --limit 25
```

The generator refuses to overwrite an existing dataset. Use `--output` for a new
version or `--resume` to reuse completed maps under an identical configuration.
Its defaults are four trajectories per map and a 0.1 m synthetic circular
robot radius, across the fixed 10/2/2 split. The existing 2x conservative map
reduction and configured 10 pixels/m are retained. There is no resizing to a
square network input; a future training loader must pad or crop aligned arrays.

Lidar uses 2,500 rays, a 20 m range, and scans about every 10 path pixels.
Navigability comes from occupancy plus footprint clearance. Annotation overlap
selects the interior navigation component and samples endpoints, without clipping
connecting routes to annotation regions. Boundary-connected exterior is excluded.
If a cropped floor opens into simulation padding and has no closed interior
component, the original image extent is used as an explicit virtual navigation
boundary. This fallback is recorded per map; occupancy inside that extent is
unchanged. It prevents leaving the map, but cannot guarantee that every free cell
inside a cropped image is indoors. Inspect routes on maps using this fallback.
Starts are seeded and preferentially spatially separated. Destinations prioritize
unobserved positions using ground-truth shortest paths. These are synthetic oracle
collection trajectories, not an evaluated autonomous exploration policy.

Snapshots are saved when observed valid-space coverage crosses 10%, 20%, ..., 90%,
and 95%. A single scan may cross several levels; it is saved once, not duplicated.
Successive saved examples must add at least 2.5 percentage points of coverage.
Exact duplicate masks within each map are suppressed. Each trajectory has a
400-scan/24-waypoint budget, so target counts are ceilings, not guaranteed totals.
Stop reasons and final coverage are recorded, including incomplete trajectories.

Each NPZ is a self-contained aligned example:

- `ground_truth`: binary occupancy, 0 free / 1 wall.
- `observation`: float32, 0 free / 1 wall / 0.5 unknown.
- `unknown_mask`: binary, 1 unknown / 0 known.
- `valid_space`: binary coverage annotation.
- `poses`: accumulated scan locations in processed map coordinates.

Known labels are copied from the target only at lidar-visible cells. This
eliminates discretization artifacts in the raycaster's labels, giving noiseless
conditional observations. Hidden cells stay exactly 0.5 with an explicit mask;
their targets must never be passed to the generator as conditioning information.
The critic's real candidate is the full target. Source-array hashes, per-trajectory
seeds, sample checksums, coverage and route metadata are saved in the manifest.

All examples inherit their map's train/validation/test split. Test data and its
review images are for data-integrity inspection only, not training or tuning.
The 14 source maps were previously designated benchmark test maps; this pilot
uses the explicitly agreed reassignment. Building grouping and overlap with the
pretrained generator's original training data remain unverified.

The review is a uniform random sample without replacement of 20% of all actual
saved examples, rounded up, using an independent fixed seed. It is not a choice
of 20% of the maps or the most attractive examples. `review/selection.json`
records every selected sample. Review PNGs show complete ground truth, partial
observation, and unknown mask; individual PNGs and six-example contact sheets
are exported. PNG observations invert occupancy for conventional display (black
wall, white free, gray unknown); use NPZ arrays as the training source.

The dataset NPZ files are excluded from Git. Reproducible source and the selected
review gallery can be committed; full generated data stays in the cloud workspace.
The published gallery is limited to 25 randomly selected examples from the full
84-example review, as requested. Its contact sheets contain only those 25 examples;
the full 20% selection remains local. Both selection seeds are recorded.
