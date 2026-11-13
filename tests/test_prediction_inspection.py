import unittest
import numpy as np

from scripts.inspect_occupancy_predictions import transition_masks,error_counts,centered_window


class PredictionInspectionTest(unittest.TestCase):
    def test_error_transitions_partition_scored_cells(self):
        target=np.array([1,1,1,1,0,0,0,0])
        before=np.array([.9,.1,.9,.1,.1,.9,.9,.1])
        after=np.array([.9,.1,.1,.9,.1,.9,.1,.9])
        domain=np.ones(8,dtype=bool)
        groups=transition_masks(before,after,target,domain)
        self.assertEqual(len(groups),8)
        self.assertTrue(all(mask.sum()==1 for mask in groups.values()))
        self.assertTrue(np.all(np.stack(list(groups.values())).sum(0)==1))
        self.assertEqual(error_counts(before,target,domain),dict(tp=2,fp=2,fn=2,tn=2))
        self.assertEqual(error_counts(after,target,domain),dict(tp=2,fp=2,fn=2,tn=2))

    def test_zooms_contain_selected_cell_and_stay_in_image(self):
        for row,col in [(0,0),(80,70),(99,119)]:
            top,left,bottom,right=centered_window(row,col,(100,120))
            self.assertTrue(0<=top<=row<bottom<=100)
            self.assertTrue(0<=left<=col<right<=120)
            self.assertEqual((bottom-top,right-left),(64,64))


if __name__=='__main__':unittest.main()
