"""Proper Bernoulli objective and streaming held-out probability metrics."""
import torch
from torch.nn import functional as F


def masked_bernoulli_loss(logits, target, unknown_mask, image_mask):
    if not (logits.shape == target.shape == unknown_mask.shape == image_mask.shape):
        raise ValueError('Logits, target, and masks must have matching shapes')
    weight = unknown_mask * image_mask
    count = weight.sum()
    if count.item() == 0:
        raise ValueError('No unknown non-padding cells to supervise')
    return (F.binary_cross_entropy_with_logits(logits, target, reduction='none')*weight).sum()/count


class OccupancyMetrics:
    def __init__(self, bins=10):
        self.bins = bins
        self.count = self.nll = self.brier = self.correct = self.tp = self.fp = self.fn = 0
        self.bin_count = torch.zeros(bins, dtype=torch.float64)
        self.bin_probability = torch.zeros(bins, dtype=torch.float64)
        self.bin_target = torch.zeros(bins, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits, target, unknown_mask, image_mask):
        eligible = (unknown_mask*image_mask).bool()
        z, y = logits[eligible].detach(), target[eligible].detach()
        if not y.numel():
            return
        p = z.sigmoid()
        self.count += y.numel()
        self.nll += F.binary_cross_entropy_with_logits(z, y, reduction='sum').item()
        self.brier += ((p-y)**2).sum().item()
        prediction = p >= .5
        self.correct += (prediction == y.bool()).sum().item()
        self.tp += (prediction & y.bool()).sum().item()
        self.fp += (prediction & ~y.bool()).sum().item()
        self.fn += (~prediction & y.bool()).sum().item()
        indices = (p*self.bins).long().clamp(max=self.bins-1).cpu()
        self.bin_count += torch.bincount(indices, minlength=self.bins).double()
        self.bin_probability.scatter_add_(0, indices, p.cpu().double())
        self.bin_target.scatter_add_(0, indices, y.cpu().double())

    def compute(self):
        if not self.count:
            return dict(cells=0)
        reliability = []
        ece = 0.
        for index in range(self.bins):
            count = int(self.bin_count[index])
            probability = self.bin_probability[index].item()/count if count else None
            frequency = self.bin_target[index].item()/count if count else None
            reliability.append(dict(lower=index/self.bins, upper=(index+1)/self.bins,
                                    cells=count, mean_probability=probability, occupied_frequency=frequency))
            if count:
                ece += count/self.count*abs(probability-frequency)
        union = self.tp+self.fp+self.fn
        return dict(cells=self.count, nll=self.nll/self.count, brier=self.brier/self.count,
                    accuracy=self.correct/self.count, occupied_iou=self.tp/union if union else None,
                    ece=ece, reliability=reliability)
