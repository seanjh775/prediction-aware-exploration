"""Validate every saved pilot example against its split and encoding contract."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def validate(output):
    manifest = json.loads((output/'manifest.json').read_text())
    splits = json.loads((output/'map_split.json').read_text())
    records = manifest['records']
    expected_maps = {map_id: split for split in ['train','validation','test'] for map_id in splits[split]}
    actual_maps = {item['map_id']: item['split'] for item in manifest['maps']}
    if expected_maps != actual_maps or len(actual_maps) != len(manifest['maps']):
        raise AssertionError('Map manifest does not match the requested split')
    if set(record['map_id'] for record in records) != set(expected_maps):
        raise AssertionError('A requested map produced no examples')
    seen_ids, seen_paths, seen_masks = set(), set(), set()
    histories = {}
    counts = {name: 0 for name in ['train','validation','test']}
    for record in records:
        if record['map_id'] not in splits[record['split']]:
            raise AssertionError('Source map assigned to wrong split')
        if record['sample_id'] in seen_ids or record['file'] in seen_paths:
            raise AssertionError('Duplicate sample identity')
        seen_ids.add(record['sample_id']); seen_paths.add(record['file'])
        with np.load(str(output/record['file'])) as data:
            gt, observation = data['ground_truth'], data['observation']
            mask, valid = data['unknown_mask'], data['valid_space']
            if not (gt.shape == observation.shape == mask.shape == valid.shape):
                raise AssertionError('Array alignment failure')
            if not all(np.isin(array, [0,1]).all() for array in [gt, mask, valid]):
                raise AssertionError('Expected binary target/mask/valid-space arrays')
            unknown = mask.astype(bool)
            if not np.all(observation[unknown] == .5):
                raise AssertionError('Hidden region leaks target labels')
            if not np.array_equal(observation[~unknown], gt[~unknown]):
                raise AssertionError('Known labels do not match target')
            coverage = ((~unknown) & (valid>0)).sum()/float(valid.sum())
            if abs(coverage-record['coverage']) > 1e-10:
                raise AssertionError('Coverage metadata mismatch')
            if hashlib.sha256(unknown.tobytes()).hexdigest() != record['unknown_mask_sha256']:
                raise AssertionError('Mask checksum mismatch')
            mask_key = (record['map_id'], record['unknown_mask_sha256'])
            if mask_key in seen_masks:
                raise AssertionError('Duplicate observation mask within a map')
            seen_masks.add(mask_key)
            key = (record['map_id'], record['trajectory'])
            if key in histories:
                old_mask, old_coverage = histories[key]
                if not np.all(unknown <= old_mask) or coverage <= old_coverage:
                    raise AssertionError('Saved trajectory does not accumulate new observations')
            histories[key] = (unknown.copy(), coverage)
        counts[record['split']] += 1
    actual_files = {p.relative_to(output).as_posix() for p in output.glob('*/*/*.npz')}
    if actual_files != seen_paths or counts != manifest['counts']:
        raise AssertionError('Manifest does not exactly cover saved samples')
    if 'config' in manifest:
        config = manifest['config']
        source = Path(config['root'])/config['dataset_root']
        for item in manifest['maps']:
            for filename, expected in item['source_sha256'].items():
                if hashlib.sha256((source/item['map_id']/filename).read_bytes()).hexdigest() != expected:
                    raise AssertionError('Source map changed after generation')
    selection = json.loads((output/'review'/'selection.json').read_text())
    if selection['total'] != len(records) or len(selection['records']) != selection['selected']:
        raise AssertionError('Review metadata mismatch')
    review_ids = [r['sample_id'] for r in selection['records']]
    if len(set(review_ids)) != len(review_ids) or not set(review_ids) <= seen_ids:
        raise AssertionError('Invalid review membership')
    from scripts.generate_masked_pilot import select_review
    if selection['records'] != select_review(records, selection['fraction'], selection['seed']):
        raise AssertionError('Review is not the declared seeded random sample')
    report = dict(status='passed', samples=len(records), counts=counts,
                  reviewed_samples=len(review_ids), maps=len(manifest['maps']),
                  checks=['source splits', 'all requested maps represented', 'unique identities', 'complete manifest',
                          'source map checksums', 'no duplicate masks within a map',
                          'binary aligned arrays', 'no hidden-label leakage', 'known-label consistency',
                          'coverage and checksums', 'monotonic accumulated observations',
                          'seeded random review membership'])
    (output/'validation_report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    print(json.dumps(validate(parser.parse_args().output), indent=2))
