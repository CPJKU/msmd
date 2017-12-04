"""This module implements aligning the score elements
to performance elements (specifically, coordinates in the MIDI
matrix).

The inputs you need for this processing are:

* MuNG file with Ly links,
* Normalized LilyPond file where the links lead,
* MIDI matrix of a performance.

On the top level, the alignment is called with a Score and a Performance.

Algorithm
---------

(...)

"""
from __future__ import print_function

import collections

import abjad
from muscima.io import parse_cropobject_list

from sheet_manager.data_model.util import SheetManagerDBError

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


def mung_midi_from_ly_links(cropobjects):
    """Adds the ``midi_pitch_code``data attribute for all CropObjects
    that have a ``ly_link`` data attribute.

    May return the CropObjects in a different order.
    """
    parser = LilyPondLinkPitchParser()
    _cdict = {c.objid: c for c in cropobjects}
    midi_pitch_codes = parser.process_cropobjects(cropobjects)
    for objid in midi_pitch_codes:
        _cdict[objid].data['midi_pitch_code'] = midi_pitch_codes[objid]

    return _cdict.values()


class LilyPondLinkPitchParser(object):
    """This is a helper class that allows interpreting backlinks
    to LilyPond files."""
    RELEVANT_CHARS = []

    # NONRELEVANT_CHARS = list("{}()[]~\\/=|<>.^!?0123456789#\"")
    NONRELEVANT_CHARS = list("{}()[]~\\/=|<>.^!?#\"")

    def __init__(self):

        self.ly_data_dict = dict()

    def process_cropobjects(self, cropobjects):
        """Processes a list of CropObjects. Returns a dict: for each ``objid``
        the ``midi_pitch_code`` integer value."""
        output = dict()
        for c in cropobjects:
            if 'ly_link' not in c.data:
                continue
            fname, row, col, _ = self.parse_ly_file_link(c.data['ly_link'])
            if fname not in self.ly_data_dict:
                self.ly_data_dict[fname] = self.load_ly_file(fname)

            token = self.ly_token_from_location(row, col,
                                                ly_data=self.ly_data_dict[fname])
            midi_pitch_code = self.ly_token_to_midi_pitch(token)
            output[c.objid] = midi_pitch_code

        self.ly_data_dict = dict()
        return output

    @staticmethod
    def load_ly_file(path):
        """Loads the LilyPond file into a lines list, so that
        it can be indexed as ``ly_data[line][column]``"""
        with open(path) as hdl:
            lines = [LilyPondLinkPitchParser.clean_line(l)
                     for l in hdl]
        return lines

    @staticmethod
    def clean_line(ly_line):
        """Clears out all the various characters we don't need
        by setting them to whitespace. Cheap and potentially very
        efficient method of cleaning the Ly file up to focus on
        pitches only.

        At the same time, the positions of retained chars must
        remain the same.
        """
        output = ly_line
        for ch in LilyPondLinkPitchParser.NONRELEVANT_CHARS:
            output = output.replace(ch, ' ')
        return output

    @staticmethod
    def ly_token_from_location(line, col, ly_data):
        """Returns the token starting at the given column on the given line
        of the ``ly_data`` lines."""
        l = ly_data[line]
        lr = l[col:]
        lr_tokens = lr.split(' ')
        tr = lr_tokens[0]

        ll = l[:col]
        ll_tokens = ll.split(' ')
        tl = ll_tokens[-1]

        t = tl + tr
        t = t.strip()
        return t

    @staticmethod
    def ly_token_to_midi_pitch(ly_token):
        """Converts the LilyPond token into the corresponding MIDI pitch
        code. Assumes the token encodes pitch absolutely."""
        note = abjad.Note(ly_token)
        wp = note.written_pitch
        midi_code = wp.number + 60
        return midi_code

    @staticmethod
    def parse_ly_file_link(link_str):
        """Parses the PDF link to the original file for a note event. This
        relies on the PDF being generated from LilyPond, with the point-and-click
        functionality enabled.

        :returns: ``(path, line, normcol, something)`` -- I don't know what the
            ``something`` part of the link is... The ``path`` is a string,
            others are ints.
        """
        protocol, path, line, normcol, something = link_str.strip().split(':')
        if path.startswith('///'):
            path = path[2:]
        line = int(line) - 1
        normcol =int(normcol)
        something = int(something)

        return path, line, normcol, something


def align_score_to_performance(score, performance):
    """For each MuNG note in the score, finds the MIDI matrix cell that
    corresponds to the onset of that note.

    :param score: A ``Score`` instance. The ``mung`` view is expected.

    :param performance: A ``Performance`` instance. The MIDI matrix feature
        must be available.

    :returns: A per-page dict of lists of ``(objid, [frame, pitch])`` tuples,
        where the ``objid`` points to the corresponding MuNG object, and
        ``[frame, pitch]`` is the frame and pitch index of the MIDI matrix
        cell that corresponds to the onset of the note encoded by this
        object.

        Note that (a) not all MIDI matrix onsets have a corresponding visual
        object, (b) not all noteheads have a corresponding onset (ties!).
    """
    score.update()
    if 'mung' not in score.views:
        raise SheetManagerDBError('Score {0}: mung view not available!'
                                  ''.format(score.name))
    mung_files = score.view_files('mung')
    mungos_per_page = []
    for f in mung_files:
        mungos = parse_cropobject_list(f)
        mungos_with_pitch = [c for c in mungos
                                  if 'midi_pitch_code' in c.data]
        mungos_per_page.append(mungos_with_pitch)

    midi_matrix = performance.load_midi_matrix()

    # Algorithm:
    #  - Create hard ordering constraints:
    #     - pages (already done: mungos_per_page)
    #     - systems
    #     - left-to-right within systems
    #     - simultaneity unrolling
    #  - Unroll MIDI matrix (unambiguous)
    #  - Align with In/Del

    raise NotImplementedError()


def group_mungos_by_system(page_mungos):
    """Groups the MuNG objects on a page into systems. Assumes
    piano music: there is a system break whenever a pitch that
    overlaps horizontally and is lower on the page is higher
    than the previous pitch.

    This method assumes no systems have been detected.
    """

    # Group symbols into columns.
    #  -
    mungos_by_left = collections.defaultdict(list)
    for m in page_mungos:
        mungos_by_left[m.left].append(m)
    rightmost_per_column = {l : max([m.right for m in mungos_by_left[l]])
                            for l in mungos_by_left}

    for l in sorted(mungos_by_left.keys()):
        pass
