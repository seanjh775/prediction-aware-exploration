"""Occupancy-based routing for synthetic observation previews."""
import numpy as np
from scipy.ndimage import distance_transform_edt, label


def navigation_region(occupancy, valid_space, radius_pixels=1.0):
    """Select an interior traversable component using annotation overlap only.

    The valid-space annotation chooses a floor component; it never clips the
    routes within that component. Border-connected free space is excluded to
    avoid selecting the free padding/exterior. Radius is a circular footprint.
    """
    if occupancy.shape != valid_space.shape or radius_pixels < 0:
        raise ValueError('Matching map shapes and nonnegative radius required')
    free = occupancy == 0
    traversable = free & (distance_transform_edt(free) > radius_pixels)
    components, _ = label(traversable)
    overlap = np.bincount(components[valid_space > 0])
    if not len(overlap):
        raise ValueError('No annotated space available')
    overlap[0] = 0
    border_ids = np.unique(np.concatenate([
        components[0], components[-1], components[:, 0], components[:, -1]]))
    overlap[border_ids[border_ids < len(overlap)]] = 0
    if overlap.max() == 0:
        raise ValueError('No interior navigable component overlaps valid space')
    return components == overlap.argmax()
