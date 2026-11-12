import unittest
from scripts.generate_masked_pilot import select_review
from scripts.export_pilot_review import limited_selection


class ReviewSelectionTest(unittest.TestCase):
    def test_reproducible_twenty_percent_without_replacement(self):
        records = [{'sample_id': index} for index in range(103)]
        selected = select_review(records, .2, 42)
        self.assertEqual(len(selected), 21)
        self.assertEqual(len({r['sample_id'] for r in selected}), 21)
        self.assertEqual(selected, select_review(records, .2, 42))
        self.assertNotEqual(selected, select_review(records, .2, 43))

    def test_publication_subset_has_only_requested_examples(self):
        records = [{'sample_id': index} for index in range(84)]
        selected = limited_selection(records, 25, 42)
        self.assertEqual(len(selected), 25)
        self.assertEqual(len({r['sample_id'] for r in selected}), 25)
        self.assertEqual(selected, limited_selection(records, 25, 42))
        self.assertTrue(all(r in records for r in selected))


if __name__ == '__main__':
    unittest.main()
