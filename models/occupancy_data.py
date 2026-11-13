"""Aligned NPZ pilot data, without resizing or hidden-label conditioning."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class OccupancyDataset(Dataset):
    def __init__(self, root, split, crop_size=0, seed=42, fixed_crops=False):
        if split not in ['train', 'validation', 'test']:
            raise ValueError('Unknown data split')
        self.root = Path(root)
        manifest = json.loads((self.root/'manifest.json').read_text())
        assignment = json.loads((self.root/'map_split.json').read_text())
        self.records = [r for r in manifest['records'] if r['split'] == split]
        if not self.records or any(r['map_id'] not in assignment[split] for r in self.records):
            raise ValueError('Empty or inconsistent source-map split')
        self.crop_size, self.seed, self.epoch = crop_size, seed, 0
        self.fixed_crops = fixed_crops

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with np.load(str(self.root/record['file'])) as data:
            arrays = {key: data[key].astype(np.float32) for key in
                      ['observation', 'unknown_mask', 'ground_truth']}
        top, left = 0, 0
        if self.crop_size:
            crop_epoch = 0 if self.fixed_crops else self.epoch
            rng = np.random.RandomState((self.seed + crop_epoch*len(self) + index) % (2**32))
            unknown = np.argwhere(arrays['unknown_mask'] > 0)
            if not len(unknown):
                raise ValueError('Example contains no unknown cells')
            anchor = unknown[rng.randint(len(unknown))]
            height, width = arrays['observation'].shape
            top = int(np.clip(anchor[0]-rng.randint(self.crop_size), 0, max(0, height-self.crop_size)))
            left = int(np.clip(anchor[1]-rng.randint(self.crop_size), 0, max(0, width-self.crop_size)))
            arrays = {key: value[top:top+self.crop_size, left:left+self.crop_size]
                      for key, value in arrays.items()}
        sample = {key: torch.from_numpy(np.ascontiguousarray(value))[None] for key, value in arrays.items()}
        sample.update(map_id=record['map_id'], sample_id=record['sample_id'], crop_origin=(top,left))
        return sample


def collate_occupancy(samples):
    # Original RGB map padding is black/free; masks mark it known, loss excludes it.
    height = ((max(s['observation'].shape[-2] for s in samples)+15)//16)*16
    width = ((max(s['observation'].shape[-1] for s in samples)+15)//16)*16
    shape = (len(samples), 1, height, width)
    batch = dict(observation=torch.zeros(shape), unknown_mask=torch.zeros(shape),
                 ground_truth=torch.zeros(shape), image_mask=torch.zeros(shape),
                 map_id=[s['map_id'] for s in samples], sample_id=[s['sample_id'] for s in samples],
                 crop_origin=[s.get('crop_origin',(0,0)) for s in samples])
    for index, sample in enumerate(samples):
        h, w = sample['observation'].shape[-2:]
        for key in ['observation', 'unknown_mask', 'ground_truth']:
            batch[key][index, :, :h, :w] = sample[key]
        batch['image_mask'][index, :, :h, :w] = 1
    return batch
