import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from models.occupancy_data import OccupancyDataset, collate_occupancy
from scripts.diagnose_occupancy_loss import group_summary


class LossDiagnosisTest(unittest.TestCase):
    def test_group_contributions_sum_to_total_loss_change(self):
        before = np.array([.1, .2, .3, .4])
        after = np.array([2., .1, .1, 3.])
        first = np.array([True, False, True, False])
        groups = [group_summary(before, after, first),group_summary(before, after, ~first)]
        self.assertAlmostEqual(sum(g['loss_sum_change'] for g in groups),float((after-before).sum()))
        self.assertEqual(sum(g['cells'] for g in groups),len(before))

    def test_crop_origin_locates_the_original_map_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = (np.indices((20,24)).sum(0)%3==0).astype(np.float32)
            unknown = np.zeros_like(target);unknown[4:17,6:22]=1
            observation=np.where(unknown,.5,target)
            np.savez_compressed(str(root/'sample.npz'),ground_truth=target,
                                unknown_mask=unknown,observation=observation)
            record=dict(sample_id='sample',map_id='a',split='train',file='sample.npz')
            (root/'manifest.json').write_text(json.dumps(dict(records=[record])))
            (root/'map_split.json').write_text(json.dumps(dict(train=['a'],validation=[],test=[])))
            sample=OccupancyDataset(root,'train',crop_size=8)[0]
            row,col=sample['crop_origin']
            self.assertTrue(np.array_equal(sample['ground_truth'][0].numpy(),target[row:row+8,col:col+8]))
            self.assertGreater(sample['unknown_mask'].sum().item(),0)
            self.assertEqual(collate_occupancy([sample])['crop_origin'],[(row,col)])


if __name__ == '__main__':
    unittest.main()
