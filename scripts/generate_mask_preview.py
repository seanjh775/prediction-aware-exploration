import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import pyastar2d
from scripts.sim_utils import Mapper, get_kth_occ_validspace_map
from scripts.map_preview_routing import navigation_region

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description='Generate an occupancy-routed lidar preview')
parser.add_argument('--robot-radius-m', type=float, default=0.1,
                    help='Synthetic circular footprint radius; default 0.1 m is a preview assumption')
args = parser.parse_args()
out = root/'work'/'masked_map_preview'
out.mkdir(parents=True, exist_ok=True)
map_id = '50052748'
assert map_id in json.loads((root/'configs/critic_map_split.json').read_text())['train']
gt, valid = get_kth_occ_validspace_map(str(root/'kth_test_maps'/map_id/'occ_map.npy'),
                                    str(root/'kth_test_maps'/map_id/'valid_space.npy'))
settings = dict(laser_range_m=20, pixel_per_meter=10, num_laser=2500, dilate_diam_for_planning=3)
mapper = Mapper(gt, settings)
safe = navigation_region(gt, valid, args.robot_radius_m * settings['pixel_per_meter'])
# Start and destination samples are annotated, but connecting routes need not be.
points = np.argwhere(safe & (valid > 0))
rng = np.random.RandomState(42)
current = points[rng.randint(len(points))]
cost = np.where(safe, 1., np.inf).astype(np.float32)
poses = [current.copy()]
for _ in range(12):
    # Oracle routing is for a visibility preview, not a deployed exploration policy.
    candidates = points[np.linalg.norm(points-current, axis=1) > 50]
    if not len(candidates):
        raise ValueError('No sufficiently distant destinations in this navigation component')
    goal = candidates[rng.randint(len(candidates))]
    path = pyastar2d.astar_path(cost, current, goal, allow_diagonal=False)
    assert path is not None
    assert np.all(safe[path[:, 0], path[:, 1]])
    poses.extend(path[10::10])
    poses.append(path[-1]); current = path[-1]
poses = np.array(poses)
indices = sorted(set([0, len(poses)//6, len(poses)//2, len(poses)-1]))
crop = (slice(500,-500), slice(500,-500))
target = gt[crop].astype(np.float32)
valid_crop = valid[crop] > 0
snapshots=[]; records=[]
for step, pose in enumerate(poses):
    mapper.observe_and_accumulate_given_pose(pose)
    if step not in indices: continue
    unknown = (mapper.obs_map[crop] == .5)
    # Exact known labels from ground truth; visibility comes from the lidar mapper.
    observation = np.where(unknown, .5, target).astype(np.float32)
    assert np.array_equal(observation[~unknown], target[~unknown])
    if snapshots: assert np.all(unknown <= snapshots[-1][1])
    index=len(snapshots)
    np.savez_compressed(str(out/('sample_%02d.npz'%index)), ground_truth=target,
                        observation=observation, unknown_mask=unknown.astype(np.uint8),
                        valid_space=valid_crop.astype(np.uint8), poses=poses[:step+1]-500)
    Image.fromarray((unknown.astype(np.uint8)*255)).save(str(out/('sample_%02d_mask.png'%index)))
    Image.fromarray(np.rint((1-observation)*255).astype(np.uint8)).save(str(out/('sample_%02d_observation.png'%index)))
    coverage=float((~unknown & valid_crop).sum()/valid_crop.sum())
    snapshots.append((observation,unknown,poses[:step+1]-500))
    records.append(dict(sample=index,scan_count=step+1,valid_space_coverage=coverage,
                        known_label_corrections=int(((mapper.obs_map[crop] != target)&~unknown).sum())))
fig, axes=plt.subplots(4,3,figsize=(15,12))
for row, ((observation,unknown,trajectory),record) in enumerate(zip(snapshots,records)):
    axes[row,0].imshow(1-target,cmap='gray',vmin=0,vmax=1)
    axes[row,1].imshow(1-observation,cmap='gray',vmin=0,vmax=1)
    axes[row,1].plot(trajectory[:,1],trajectory[:,0],color='#2b83ff',lw=1)
    axes[row,1].scatter(trajectory[-1,1],trajectory[-1,0],c='#ff5722',s=18)
    axes[row,2].imshow(unknown,cmap='gray',vmin=0,vmax=1)
    axes[row,0].set_title('Complete ground truth (training target)')
    axes[row,1].set_title('Observed: %.1f%% of valid space | %d scans'%(100*record['valid_space_coverage'],record['scan_count']))
    axes[row,2].set_title('Unknown mask: white = hidden')
    for ax in axes[row]: ax.axis('off')
fig.suptitle('Training map 50052748 | accumulated lidar observations\nBlack = wall, white = free, gray = unknown; blue = trajectory, orange = robot',fontsize=14)
fig.subplots_adjust(top=.90, bottom=.02, left=.01, right=.99, hspace=.25, wspace=.04)
fig.savefig(str(out/'preview.png'),dpi=140)
Image.fromarray(np.rint((1-target)*255).astype(np.uint8)).save(str(out/'ground_truth.png'))
metadata=dict(map_id=map_id,split='train',seed=42,lidar=settings,
              shape=list(target.shape),occupancy_encoding='0 free, 1 occupied; observation unknown=0.5',
              mask_encoding='1 unknown, 0 known',routing='oracle occupancy-based shortest paths; valid space used for endpoint sampling and coverage only',
              robot_radius_m=args.robot_radius_m,
              robot_footprint_note='Synthetic circular footprint assumption, not a measured robot specification',
              navigation_cells=int(safe.sum()),
              route_scans_outside_valid_space=int((valid[poses[:,0],poses[:,1]] == 0).sum()),
              known_labels='ground-truth labels at lidar-visible cells',samples=records)
(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(metadata,indent=2))
