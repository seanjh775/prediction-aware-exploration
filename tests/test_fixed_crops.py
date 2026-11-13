import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from models.occupancy_data import OccupancyDataset
from scripts.train_probabilistic_lama import select_sample_ids


class FixedCropTest(unittest.TestCase):
    def test_fixed_crops_repeat_across_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            target=np.random.RandomState(0).randint(0,2,(48,60)).astype(np.float32)
            np.savez_compressed(str(root/'a.npz'),ground_truth=target,unknown_mask=np.ones_like(target),observation=np.full_like(target,.5))
            (root/'manifest.json').write_text(json.dumps(dict(records=[dict(sample_id='a',map_id='a',split='train',file='a.npz')])))
            (root/'map_split.json').write_text(json.dumps(dict(train=['a'],validation=[],test=[])))
            fixed=OccupancyDataset(root,'train',16,fixed_crops=True)
            first=fixed[0];fixed.epoch=7;later=fixed[0]
            self.assertEqual(first['crop_origin'],later['crop_origin'])
            for key in ['observation','unknown_mask','ground_truth']:
                self.assertTrue(torch.equal(first[key],later[key]))
            changing=OccupancyDataset(root,'train',16)
            first=changing[0];changing.epoch=7;later=changing[0]
            self.assertNotEqual(first['crop_origin'],later['crop_origin'])

    def test_explicit_ids_cannot_cross_split_or_duplicate(self):
        dataset=SimpleNamespace(records=[dict(sample_id='train_a'),dict(sample_id='train_b')])
        with self.assertRaises(ValueError):select_sample_ids(dataset,['validation_a'])
        with self.assertRaises(ValueError):select_sample_ids(dataset,['train_a','train_a'])
        select_sample_ids(dataset,['train_b','train_a'])
        self.assertEqual([r['sample_id'] for r in dataset.records],['train_b','train_a'])


if __name__ == '__main__':unittest.main()
