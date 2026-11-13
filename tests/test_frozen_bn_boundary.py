import unittest
import numpy as np
import torch

from models.occupancy_objective import boundary_counts,OccupancyMetrics
from scripts.train_probabilistic_lama import set_training_mode


class FrozenBNAndBoundaryTest(unittest.TestCase):
    def test_statistics_stay_fixed_while_affine_parameters_learn(self):
        model=torch.nn.Sequential(torch.nn.BatchNorm2d(2))
        before={k:v.clone() for k,v in model.state_dict().items()}
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        for _ in range(2):
            set_training_mode(model,True)
            self.assertTrue(model.training)
            self.assertFalse(model[0].training)
            optimizer.zero_grad()
            model(torch.randn(2,2,6,6)+3).square().mean().backward()
            self.assertIsNotNone(model[0].weight.grad)
            optimizer.step()
        for key in ['0.running_mean','0.running_var','0.num_batches_tracked']:
            self.assertTrue(torch.equal(before[key],model.state_dict()[key]))
        self.assertFalse(torch.equal(before['0.weight'],model[0].weight))

    def test_one_pixel_shift_matches_but_two_pixel_shift_does_not(self):
        target=np.zeros((20,20),dtype=bool);target[4:16,8]=True
        domain=np.ones_like(target)
        shifted=np.roll(target,1,axis=1)
        counts=boundary_counts(shifted,target,domain,domain,1)
        self.assertEqual(counts['matched_predicted'],counts['predicted'])
        self.assertEqual(counts['matched_actual'],counts['actual'])
        self.assertEqual((shifted&target).sum(),0)
        far=boundary_counts(np.roll(target,2,axis=1),target,domain,domain,1)
        self.assertEqual(far['matched_predicted'],0)
        self.assertEqual(far['matched_actual'],0)

    def test_padding_and_known_cells_do_not_create_boundary_credit(self):
        target=np.ones((10,10),dtype=bool)
        domain=np.ones_like(target)
        counts=boundary_counts(target,target,domain,domain,1)
        self.assertEqual(counts['actual'],0)  # excludes artificial image perimeter
        unknown=np.zeros_like(domain);unknown[2:8,2:8]=True
        counts=boundary_counts(target,target,unknown,domain,1)
        self.assertEqual(counts['actual'],0)  # unknown-mask perimeter is not a wall boundary
        target[:]=False;target[3:7,5]=True
        counts=boundary_counts(target,target,np.zeros_like(domain),domain,1)
        self.assertEqual(counts['actual'],0)

    def test_metric_reports_precision_recall_and_f1(self):
        target=torch.zeros(1,1,20,20);target[:,:,4:16,8]=1
        prediction=torch.roll(target,1,dims=3)
        logits=prediction*20-10
        metric=OccupancyMetrics();metric.update(logits,target,torch.ones_like(target),torch.ones_like(target))
        result=metric.compute()
        self.assertEqual(result['boundary_f1'],1)
        self.assertEqual(result['boundary_precision'],1)
        self.assertEqual(result['boundary_recall'],1)
        self.assertEqual(result['occupied_iou'],0)


if __name__ == '__main__':
    unittest.main()
