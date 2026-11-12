import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.validate_masked_pilot import validate


class PilotValidationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root/'train'/'map_a').mkdir(parents=True)
        (self.root/'review').mkdir()
        self.mask = np.array([[0, 1], [0, 1]], dtype=np.uint8)
        self.gt = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        self.observation = np.where(self.mask, .5, self.gt).astype(np.float32)
        self.record = dict(sample_id='a', file='train/map_a/a.npz', split='train', map_id='map_a',
                           trajectory=0, coverage=.5,
                           unknown_mask_sha256=hashlib.sha256(self.mask.astype(bool).tobytes()).hexdigest())
        self.save_arrays()
        (self.root/'map_split.json').write_text(json.dumps(dict(train=['map_a'], validation=[], test=[])))
        (self.root/'manifest.json').write_text(json.dumps(dict(records=[self.record], maps=[dict(map_id='map_a', split='train')],
            counts=dict(train=1, validation=0, test=0))))
        (self.root/'review'/'selection.json').write_text(json.dumps(dict(
            total=1, selected=1, seed=42, fraction=.2, records=[self.record])))

    def tearDown(self):
        self.directory.cleanup()

    def save_arrays(self):
        np.savez_compressed(str(self.root/self.record['file']), ground_truth=self.gt,
                            observation=self.observation, unknown_mask=self.mask,
                            valid_space=np.ones_like(self.mask))

    def test_accepts_aligned_masked_example(self):
        self.assertEqual(validate(self.root)['status'], 'passed')

    def test_rejects_hidden_ground_truth_leakage(self):
        self.observation[self.mask.astype(bool)] = self.gt[self.mask.astype(bool)]
        self.save_arrays()
        with self.assertRaisesRegex(AssertionError, 'leaks target'):
            validate(self.root)

    def test_rejects_wrong_map_split(self):
        (self.root/'map_split.json').write_text(json.dumps(dict(train=[], validation=['map_a'], test=[])))
        with self.assertRaisesRegex(AssertionError, 'requested split'):
            validate(self.root)


if __name__ == '__main__':
    unittest.main()
