"""Proper Bernoulli objective and streaming held-out probability metrics."""
import torch
from torch.nn import functional as F


def boundary_counts(prediction, target, eligible, image_support, tolerance=1.):
    """Micro-count boundary matches within Euclidean pixel distance (evaluation only)."""
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
    if tolerance < 0:
        raise ValueError('Boundary tolerance must be nonnegative')
    neighborhood = generate_binary_structure(2,1)
    # Extract physical boundaries before applying unknown-cell eligibility. This
    # avoids inventing edges along the unknown/known mask. Exclude image edges
    # where neighbors fall outside real image support.
    interior = binary_erosion(image_support, structure=neighborhood, border_value=0)
    domain = eligible & interior
    pred_boundary = prediction & ~binary_erosion(prediction, structure=neighborhood, border_value=0) & domain
    true_boundary = target & ~binary_erosion(target, structure=neighborhood, border_value=0) & domain
    predicted, actual = int(pred_boundary.sum()),int(true_boundary.sum())
    matched_pred = int((pred_boundary & (distance_transform_edt(~true_boundary)<=tolerance)).sum()) if actual else 0
    matched_true = int((true_boundary & (distance_transform_edt(~pred_boundary)<=tolerance)).sum()) if predicted else 0
    return dict(predicted=predicted, actual=actual, matched_predicted=matched_pred, matched_actual=matched_true)


def masked_bernoulli_loss(logits, target, unknown_mask, image_mask):
    if not (logits.shape == target.shape == unknown_mask.shape == image_mask.shape):
        raise ValueError('Logits, target, and masks must have matching shapes')
    weight = unknown_mask * image_mask
    count = weight.sum()
    if count.item() == 0:
        raise ValueError('No unknown non-padding cells to supervise')
    return (F.binary_cross_entropy_with_logits(logits, target, reduction='none')*weight).sum()/count


class OccupancyMetrics:
    def __init__(self, bins=10, boundary_tolerance=1.):
        self.bins = bins
        self.count = self.nll = self.brier = self.correct = self.tp = self.fp = self.fn = 0
        self.bin_count = torch.zeros(bins, dtype=torch.float64)
        self.bin_probability = torch.zeros(bins, dtype=torch.float64)
        self.bin_target = torch.zeros(bins, dtype=torch.float64)
        self.boundary_tolerance = boundary_tolerance
        self.boundary = dict(predicted=0, actual=0, matched_predicted=0, matched_actual=0)

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
        # Known neighbors use their exact observed labels, while only unknown
        # real-image boundary pixels contribute to the reported scores.
        completed = torch.where(eligible, logits>=0, target.bool()).cpu().numpy()
        truth, support, domain = target.bool().cpu().numpy(), image_mask.bool().cpu().numpy(), eligible.cpu().numpy()
        for index in range(logits.shape[0]):
            counts = boundary_counts(completed[index,0],truth[index,0],domain[index,0],support[index,0],self.boundary_tolerance)
            for key,value in counts.items():
                self.boundary[key] += value

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
        b = self.boundary
        if b['predicted'] or b['actual']:
            precision = b['matched_predicted']/b['predicted'] if b['predicted'] else 0.
            recall = b['matched_actual']/b['actual'] if b['actual'] else 0.
            f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
        else:
            precision = recall = f1 = None
        return dict(cells=self.count, nll=self.nll/self.count, brier=self.brier/self.count,
                    accuracy=self.correct/self.count, occupied_iou=self.tp/union if union else None,
                    ece=ece, reliability=reliability, boundary_tolerance_pixels=self.boundary_tolerance,
                    boundary_precision=precision, boundary_recall=recall, boundary_f1=f1,
                    boundary_counts=dict(b))
