"""Generate split-safe, coverage-sampled lidar observations and a random review."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def select_review(records, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError('Review fraction must be in (0, 1]')
    count = int(math.ceil(len(records) * fraction))
    indices = np.random.RandomState(seed).choice(len(records), count, replace=False)
    return [records[index] for index in sorted(indices.tolist())]


def generate_map(job):
    import pyastar2d
    from scripts.sim_utils import Mapper, get_kth_occ_validspace_map
    from scripts.map_preview_routing import navigation_region

    root, output = Path(job['root']), Path(job['output'])
    map_id, split = job['map_id'], job['split']
    directory = output / split / map_id
    if (directory/'map_manifest.json').exists():
        result = json.loads((directory/'map_manifest.json').read_text())
        print('Reusing completed map '+map_id, flush=True)
        return result
    # Preserve the exact existing map preprocessing and simulator padding.
    map_dir = root / job['dataset_root'] / map_id
    gt, valid = get_kth_occ_validspace_map(str(map_dir/'occ_map.npy'), str(map_dir/'valid_space.npy'))
    boundary = 'closed_occupancy_component'
    try:
        safe = navigation_region(gt, valid, job['robot_radius_m'] * job['lidar']['pixel_per_meter'])
    except ValueError as error:
        if 'No interior navigable component' not in str(error):
            raise
        domain = np.zeros_like(gt, dtype=bool)
        domain[500:-500, 500:-500] = True
        safe = navigation_region(gt, valid, job['robot_radius_m'] * job['lidar']['pixel_per_meter'],
                                 navigation_domain=domain)
        boundary = 'original_map_extent_virtual_boundary'
    points = np.argwhere(safe & (valid > 0))
    if not len(points):
        raise ValueError('No valid starting points for ' + map_id)
    crop = (slice(500, -500), slice(500, -500))
    target = gt[crop].astype(np.uint8)
    valid_crop = (valid[crop] > 0).astype(np.uint8)
    valid_count = int(valid_crop.sum())
    if not valid_count:
        raise ValueError('Empty coverage denominator for ' + map_id)
    cost = np.where(safe, 1., np.inf).astype(np.float32)
    seed = (job['seed'] + int(hashlib.sha256(map_id.encode()).hexdigest()[:8], 16)) % (2**32)
    directory.mkdir(parents=True, exist_ok=True)
    starts, records, trajectories = [], [], []
    fingerprints = set()
    source_hashes = {name: hashlib.sha256((map_dir/name).read_bytes()).hexdigest()
                     for name in ['occ_map.npy', 'valid_space.npy']}

    for trajectory in range(job['trajectories_per_map']):
        trajectory_seed = (seed + trajectory) % (2**32)
        rng = np.random.RandomState(trajectory_seed)
        # Prefer spatially separated starts while retaining seeded randomness.
        candidates = points
        if starts:
            distance = np.min(np.stack([np.linalg.norm(points-p, axis=1) for p in starts]), axis=0)
            candidates = points[distance >= min(50, distance.max())]
        current = candidates[rng.randint(len(candidates))].copy()
        starts.append(current.copy())
        mapper = Mapper(gt, job['lidar'])
        visited_poses, coverage_history = [], []
        level = 0
        last_saved = -1.
        trajectory_records = []
        invalid_label_count = 0

        def observe(pose):
            nonlocal level, last_saved, invalid_label_count
            assert safe[pose[0], pose[1]]
            mapper.observe_and_accumulate_given_pose(pose)
            visited_poses.append(pose.copy())
            unknown = mapper.obs_map[crop] == .5
            coverage = float(((~unknown) & (valid_crop > 0)).sum() / valid_count)
            if coverage_history and coverage < coverage_history[-1]:
                raise AssertionError('Accumulated coverage decreased')
            coverage_history.append(coverage)
            if level >= len(job['coverage_targets']) or coverage < job['coverage_targets'][level]:
                return
            crossed = []
            while level < len(job['coverage_targets']) and coverage >= job['coverage_targets'][level]:
                crossed.append(job['coverage_targets'][level]); level += 1
            # One example per scan, even when a scan crosses several thresholds.
            if coverage - last_saved < job['minimum_coverage_gain']:
                return
            fingerprint = hashlib.sha256(unknown.tobytes()).hexdigest()
            if fingerprint in fingerprints:
                return
            fingerprints.add(fingerprint)
            last_saved = coverage
            sample_id = '{}_t{:02d}_s{:02d}'.format(map_id, trajectory, len(trajectory_records))
            relative = Path(split)/map_id/(sample_id+'.npz')
            observation = np.where(unknown, .5, target).astype(np.float32)
            if not np.array_equal(observation[~unknown], target[~unknown]):
                raise AssertionError('Known observation differs from target')
            poses = np.asarray(visited_poses, dtype=np.int32)
            invalid_label_count = int(((mapper.obs_map[crop] != target) & ~unknown).sum())
            np.savez_compressed(str(output/relative), ground_truth=target,
                                observation=observation, unknown_mask=unknown.astype(np.uint8),
                                valid_space=valid_crop, poses=poses-500)
            record = dict(sample_id=sample_id, split=split, map_id=map_id,
                          trajectory=trajectory, seed=trajectory_seed, file=relative.as_posix(),
                          scans=len(visited_poses), coverage=coverage, thresholds_crossed=crossed,
                          shape=list(target.shape), unknown_mask_sha256=fingerprint,
                          known_label_corrections=invalid_label_count,
                          route_scans_outside_valid_space=int((valid[poses[:,0],poses[:,1]] == 0).sum()))
            trajectory_records.append(record); records.append(record)

        observe(current)
        reason = 'waypoint_budget'
        for waypoint in range(job['max_waypoints']):
            if level == len(job['coverage_targets']):
                reason = 'coverage_target'; break
            if len(visited_poses) >= job['max_scans']:
                reason = 'scan_budget'; break
            distances = np.linalg.norm(points-current, axis=1)
            candidates = points[distances > job['minimum_goal_distance_pixels']]
            if not len(candidates):
                reason = 'no_distant_goal'; break
            # Oracle synthetic collection: prioritize still-unobserved destinations.
            unseen = mapper.obs_map[candidates[:,0], candidates[:,1]] == .5
            if np.any(unseen):
                candidates = candidates[unseen]
            goal = candidates[rng.randint(len(candidates))]
            path = pyastar2d.astar_path(cost, current, goal, allow_diagonal=False)
            if path is None or not np.all(safe[path[:,0],path[:,1]]):
                raise AssertionError('Invalid planned path on ' + map_id)
            sampled = path[job['scan_stride_pixels']::job['scan_stride_pixels']]
            if not len(sampled) or not np.array_equal(sampled[-1], path[-1]):
                sampled = np.concatenate([sampled, path[-1:]], axis=0)
            for pose in sampled:
                observe(pose)
                if level == len(job['coverage_targets']) or len(visited_poses) >= job['max_scans']:
                    break
            current = visited_poses[-1]
        if level == len(job['coverage_targets']):
            reason = 'coverage_target'
        elif len(visited_poses) >= job['max_scans']:
            reason = 'scan_budget'
        trajectories.append(dict(trajectory=trajectory, seed=trajectory_seed,
                                 start=(starts[-1]-500).tolist(), scans=len(visited_poses),
                                 final_coverage=coverage_history[-1], samples=len(trajectory_records),
                                 stop_reason=reason))
        del mapper
    result = dict(map_id=map_id, split=split, source_sha256=source_hashes,
                  navigation_boundary=boundary,
                  trajectories=trajectories, records=records)
    save_json(directory/'map_manifest.json', result)
    print('{} {}: {} samples; final coverage {}'.format(
        split, map_id, len(records), ', '.join('{:.1%}'.format(t['final_coverage']) for t in trajectories)), flush=True)
    return result


def render_review(output, records, fraction, seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    selected = select_review(records, fraction, seed)
    directory = output/'review'
    directory.mkdir(exist_ok=True)
    for record in selected:
        with np.load(str(output/record['file'])) as data:
            stem = record['sample_id']
            for name, array in [('mask', data['unknown_mask']*255),
                                ('observation', np.rint((1-data['observation'])*255)),
                                ('ground_truth', (1-data['ground_truth'])*255)]:
                Image.fromarray(array.astype(np.uint8)).save(str(directory/(stem+'_'+name+'.png')))
    pages = []
    for start in range(0, len(selected), 6):
        batch = selected[start:start+6]
        figure, axes = plt.subplots(len(batch), 3, figsize=(15, 2.5*len(batch)), squeeze=False)
        for row, record in enumerate(batch):
            with np.load(str(output/record['file'])) as data:
                arrays = [1-data['ground_truth'], 1-data['observation'], data['unknown_mask']]
                titles = ['Ground truth', 'Observation | {:.1%} coverage'.format(record['coverage']), 'Mask: white = unknown']
                for col in range(3):
                    axes[row,col].imshow(arrays[col], cmap='gray', vmin=0, vmax=1)
                    axes[row,col].set_title(titles[col], fontsize=9)
                    axes[row,col].axis('off')
                axes[row,0].text(0, -.12, record['split']+' | '+record['sample_id'],
                                 transform=axes[row,0].transAxes, fontsize=8)
        figure.suptitle('Seeded random 20% review | page {}'.format(len(pages)+1))
        figure.tight_layout(rect=(0,.01,1,.97), h_pad=2)
        name = 'contact_sheet_{:02d}.png'.format(len(pages)+1)
        figure.savefig(str(directory/name), dpi=130); plt.close(figure)
        pages.append(name)
    save_json(directory/'selection.json', dict(seed=seed, fraction=fraction, total=len(records),
                                               selected=len(selected), pages=pages, records=selected))
    return len(selected), pages


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root/'data'/'kth_masked_pilot_v1')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--trajectories-per-map', type=int, default=4)
    parser.add_argument('--resume', action='store_true', help='Reuse completed maps with matching configuration')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise ValueError('Choose a new output directory; refusing to overwrite a dataset')
    output.mkdir(parents=True, exist_ok=True)
    split = json.loads((root/'configs/critic_map_split.json').read_text())
    config = dict(root=str(root), output=str(output), dataset_root=split['dataset_root'],
                  seed=42, robot_radius_m=.1, trajectories_per_map=args.trajectories_per_map,
                  coverage_targets=[.1,.2,.3,.4,.5,.6,.7,.8,.9,.95], minimum_coverage_gain=.025,
                  max_waypoints=24, max_scans=400, scan_stride_pixels=10,
                  minimum_goal_distance_pixels=50,
                  lidar=dict(laser_range_m=20, pixel_per_meter=10, num_laser=2500,
                             dilate_diam_for_planning=3))
    ids = [map_id for name in ['train','validation','test'] for map_id in split[name]]
    if len(ids) != len(set(ids)):
        raise ValueError('Map split overlap detected')
    if args.resume and (output/'config.json').exists():
        if json.loads((output/'config.json').read_text()) != config:
            raise ValueError('Cannot resume with a different generation configuration')
    save_json(output/'config.json', config)
    save_json(output/'map_split.json', split)
    jobs = [dict(config, map_id=map_id, split=name)
            for name in ['train','validation','test'] for map_id in split[name]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(generate_map, jobs))
    records = [record for result in results for record in result['records']]
    if not records:
        raise ValueError('No examples generated')
    manifest = dict(version=1, config=config, maps=results, records=records,
                    counts={name:sum(r['split']==name for r in records) for name in ['train','validation','test']},
                    caveats=['Oracle routes and noiseless ground-truth labels at lidar-visible cells.',
                             'Full processed map resolution retained; no resize/crop augmentation.',
                             'Test examples are inspection-only and must not enter training or tuning.'])
    save_json(output/'manifest.json', manifest)
    count, pages = render_review(output, records, .2, 20261112)
    print(json.dumps(dict(counts=manifest['counts'], total=len(records), review_count=count,
                          review_pages=len(pages), output=str(output))), flush=True)


if __name__ == '__main__':
    main()
