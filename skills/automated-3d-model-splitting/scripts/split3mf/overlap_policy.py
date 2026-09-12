"""Physical assembly acceptance, separate from Boolean numerical precision."""
import math


DEFAULT_IGNORE_OVERLAP_RATIO = 0.01


def validate_ignore_overlap_ratio(ratio):
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('ignored overlap ratio must be finite and between 0 and 1')


def ratio_volume_limit(cutting_volume_mm3, ratio):
    validate_ignore_overlap_ratio(ratio)
    if cutting_volume_mm3 is None:
        return 0.0
    validate_ignore_overlap_threshold(cutting_volume_mm3)
    return cutting_volume_mm3 * ratio


def ratio_measurement(volume, cutting_volume_mm3, ratio):
    limit = ratio_volume_limit(cutting_volume_mm3, ratio)
    return dict(cutting_volume_mm3=cutting_volume_mm3,
                overlap_cutting_volume_ratio=(volume / cutting_volume_mm3
                    if cutting_volume_mm3 else None),
                ignore_overlap_ratio=ratio,
                ignored_by_cut_volume_ratio=overlap_is_ignored(volume, limit))


def validate_ignore_overlap_threshold(threshold):
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError('ignored overlap threshold must be finite and nonnegative')


def overlap_is_ignored(volume, threshold):
    """Compare the summed intersection volume of one pair, not each fragment."""
    validate_ignore_overlap_threshold(threshold)
    # Boolean summation can report an exact 1 as 0.9999999999999998.
    # Treat machine-near-boundary values conservatively as the boundary;
    # this guard never expands the physical acceptance range.
    return bool(0 <= volume < threshold
                and not math.isclose(volume, threshold, rel_tol=1e-12, abs_tol=0.0))
