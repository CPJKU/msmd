
import numpy as np
import matplotlib.path as mplPath


# init color printer
class BColors:
    """
    Colored command line output formatting
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def __init__(self):
        """ Constructor """
        pass

    def print_colored(self, string, color):
        """ Change color of string """
        return color + string + BColors.ENDC


def get_target_shape(img, target_width):
    """Given a target image width, compute the target height for resizing
    the given image without changing its aspect ratio.

    :returns: ``(target_height, target_width)`` in integers.
    """
    ratio = float(target_width) / img.shape[1]
    target_height = img.shape[0] * ratio

    return int(target_height), int(target_width)


def sort_by_rows(coords, start_pos, window=40, window_top=None, window_bottom=None):
    """
    group annotations by rows
    """

    if window_top is None and window_bottom is None:
        window_top = window_bottom = window

    row_sorted_coords = np.zeros((0, 2), dtype=np.float32)
    for start in start_pos:
        idx1 = np.nonzero(coords[:, 0] > (start[0] - window_top))[0]
        idx2 = np.nonzero(coords[:, 0] < (start[0] + window_bottom))[0]
        idxs = np.intersect1d(idx1, idx2)
        rc = coords[idxs]
        sorted_idx = np.argsort(rc[:, 1])
        rc = rc[sorted_idx]
        row_sorted_coords = np.vstack((row_sorted_coords, rc))

    return start_pos, row_sorted_coords


def sort_by_roi(coords, rois):
    """
    group annotations by system, if systems are available.
    """

    # !!!!!!!!!!!!!!
    # This is where the noteheads are aligned against the MIDI.
    # The coordinates of noteheads are sorted into systems
    # and left-to-right within a system.
    # The coordinates of onsets are simply taken from the MIDI ordering.
    row_sorted_coords = np.zeros((0, 2), dtype=np.float32)

    for roi in rois:

        # initialize bounding box
        bbPath = mplPath.Path(roi)

        # get notes inside bounding box
        idxs = bbPath.contains_points(coords)

        # sort coordinates of row
        rc = coords[idxs]
        sorted_idx = np.argsort(rc[:, 1])
        rc = rc[sorted_idx]
        row_sorted_coords = np.vstack((row_sorted_coords, rc))

    return row_sorted_coords


def natsort(l):
    """ natural sorting of file name strings """
    l = np.asarray(l)
    print l
    if len(l) > 1:
        sorted_idx = np.argsort([int(s.split("-")[1].split(".")[0]) for s in l])
        print sorted_idx
        return l[sorted_idx]
    else:
        return l


def greater_than_zero_intervals(a):
    """Find nonzero interval bounds on array ``a``.

    ``stackoverflow.com/questions/28777795/quickly-find-non-zero-intervals``

    """
    isntzero = np.concatenate(([0], np.greater(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isntzero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def corners2bbox(corners):
    """Convert the corners representation of a region to
    a bounding box."""
    t, l = corners[0]
    b, r = corners[2]
    return t, l, b, r