import unittest
from types import SimpleNamespace
from scripts.train_probabilistic_lama import select_training_maps, select_validation_examples


class TrainingSubsetTest(unittest.TestCase):
    def test_training_filter_rejects_held_out_map(self):
        dataset = SimpleNamespace(records=[dict(map_id='train_a'),dict(map_id='train_b')])
        with self.assertRaises(ValueError):
            select_training_maps(dataset,['validation_a'])
        select_training_maps(dataset,['train_b'])
        self.assertEqual(dataset.records,[dict(map_id='train_b')])

    def test_validation_selection_covers_each_map_and_coverage_range(self):
        dataset = SimpleNamespace(records=[dict(map_id=map_id,coverage=i/10)
                                          for map_id in ['a','b'] for i in range(9)])
        select_validation_examples(dataset,3)
        self.assertEqual(len(dataset.records),6)
        for map_id in ['a','b']:
            self.assertEqual([r['coverage'] for r in dataset.records if r['map_id']==map_id],[0,.4,.8])


if __name__ == '__main__':
    unittest.main()
