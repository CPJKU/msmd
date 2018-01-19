
from __future__ import print_function

import collections

import numpy as np
import matplotlib.path as mplPath


# init color printer
import yaml


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
    print(l)
    if len(l) > 1:
        sorted_idx = np.argsort([int(s.split("-")[1].split(".")[0]) for s in l])
        print(sorted_idx)
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


def aggregate_dicts(dicts):
    output = collections.defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            output[k].append(v)
    return output


def reduce_dicts(dicts, fn, **fn_kwargs):
    aggregate = aggregate_dicts(dicts)
    average = {k: fn(v, **fn_kwargs)
               for k, v in aggregate.items()}
    return average


def msmd_stats_latex(msmd, split_file, split_name, caption=None, label=None):
    train_stats, valid_stats, test_stats = msmd.stats_on_split(split_file)

    stats = {'train': train_stats, 'valid': valid_stats, 'test': test_stats}

    with open(split_file, 'rb') as hdl:
        split = yaml.load(hdl)
    n_train_pieces = len(split['train'])
    n_valid_pieces = len(split['valid'])
    n_test_pieces = len(split['test'])
    n_pieces = {'train': n_train_pieces,
                'valid': n_valid_pieces,
                'test': n_test_pieces}

    n_total_pieces = n_train_pieces + n_valid_pieces + n_test_pieces
    n_total_aln = train_stats['n_aln_pairs'] \
                  + valid_stats['n_aln_pairs'] \
                  + test_stats['n_aln_pairs']

    # Table header
    lines = list()
    lines.append('\\begin{table}[ht]')
    lines.append('\t\\begin{center}')
    lines.append('\t\\begin{tabular}{rcrcccc}')
    lines.append('\t\\toprule')
    lines.append('\t\\textbf{Split Name} &'
                 ' \\textbf{\\# Pieces/Aln. Pairs} &'
                 ' \\textbf{Part} &'
                 ' \\textbf{\\# Pieces} &'
                 ' \\textbf{\\# Noteheads} &'
                 ' \\textbf{\\# Events} &'
                 ' \\textbf{\\# Aln. Pairs} \\\\')
    lines.append('\t\\midrule')

    # Stats lines
    for line_name in ('train', 'valid', 'test'):
        current_stats = stats[line_name]

        fields = []
        # First line for given split has the split header fields
        if line_name == 'train':
            fields.append(split_name)
            fields.append('{0} / {1}'.format(n_total_pieces, n_total_aln))
        else:
            fields.extend(['', ''])

        fields.append(line_name)

        fields.append(n_pieces[line_name])
        n_noteheads = current_stats['n_mungos'] - current_stats['n_system_mungos']
        fields.append(n_noteheads)
        fields.append(current_stats['n_events'])
        fields.append(current_stats['n_aln_pairs'])

        line = '\t' + ' & '.join(map(str, fields))
        line += ' \\\\'

        lines.append(line)
    lines.append('\t\\bottomrule')

    # Table footer
    lines.append('\t\\end{tabular}')
    lines.append('\t\\end{center}')
    if caption is not None:
        lines.append('\t\\caption{{0}}'.format(caption))
    if label is not None:
        lines.append('\t\\label{{0}}'.format(label))
    lines.append('\\end{table}')

    return '\n'.join(lines)
