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
import copy
import logging
import pprint
import string

import abjad
import numpy

from skimage.measure import regionprops

from muscima.cropobject import cropobjects_merge_bbox, CropObject, link_cropobjects
from muscima.graph import NotationGraph
from muscima.inference_engine_constants import InferenceEngineConstants
from muscima.io import parse_cropobject_list

from sheet_manager.data_model.score import group_mungos_by_column
from sheet_manager.data_model.util import SheetManagerDBError
from sheet_manager.midi_parser import notes_to_onsets, FPS
from sheet_manager.utils import greater_than_zero_intervals

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


class SheetManagerLyParsingError(Exception):
    pass


def mung_midi_from_ly_links(cropobjects):
    """Adds the ``midi_pitch_code``data attribute for all CropObjects
    that have a ``ly_link`` data attribute.

    May return the CropObjects in a different order.
    """
    parser = LilyPondLinkPitchParser()
    _cdict = {c.objid: c for c in cropobjects}
    midi_pitch_codes = parser.process_mungos(cropobjects)
    for objid in midi_pitch_codes:
        _cdict[objid].data['midi_pitch_code'] = midi_pitch_codes[objid]

    return _cdict.values()


class LilyPondLinkPitchParser(object):
    """This is a helper class that allows interpreting backlinks
    to LilyPond files."""
    RELEVANT_CHARS = []

    # NONRELEVANT_CHARS = list("{}()[]_~\\/=|<>.^!?0123456789#\"")
    # ...this one was deprecated, since we (a) need ties (~), (b)
    #    don't have to strip out durations.
    NONRELEVANT_CHARS = list("{}()[]\\_/=|<>.^!?#\"%-\t0123456789")

    STRICT_NONRELEVANT_CHARS = list("{}()[]\\_/=|<>.^!?#\"%-~\t0123456789")

    TIE_CHAR = '~'

    def __init__(self):

        self.ly_data_dict = dict()
        self.ly_token_dict = dict()

    def process_mungos(self, mungos):
        """Processes a list of MuNG objects. Returns a dict: for each ``objid``
        the ``midi_pitch_code`` integer value.

        For each MuNG object that is tied, adds the ``tied=True`` attribute
        to its ``data``.
        """
        output = dict()
        _n_ties = 0
        for c in mungos:
            if 'ly_link' not in c.data:
                continue
            fname, row, col, _ = self.parse_ly_file_link(c.data['ly_link'])
            if fname not in self.ly_data_dict:
                lines, tokens = self.load_ly_file(fname, with_tokens=True)
                self.ly_data_dict[fname] = lines
                self.ly_token_dict[fname] = tokens

            if self.check_tie_before_location(row, col, ly_data=self.ly_data_dict[fname]):
                logging.info('PROCESS FOUND TIE: mungo objid = {0},'
                             ' location = {1}'.format(c.objid, (c.top, c.left)))
                _n_ties += 1
                c.data['tied'] = 1
            else:
                c.data['tied'] = 0

            token = self.ly_token_from_location(row, col,
                                                ly_data=self.ly_data_dict[fname])

            try:
                midi_pitch_code = self.ly_token_to_midi_pitch(token)
            except SheetManagerLyParsingError as e:
                raise SheetManagerLyParsingError('Token {0} at location {1}'
                                                 ' could not be parsed. Line:'
                                                 ' {2}'
                                                 ''.format(token, (row, col),
                                                           self.ly_data_dict[fname][row]))
            output[c.objid] = midi_pitch_code

        logging.info('TOTAL TIED NOTES: {0}'.format(_n_ties))

        self.ly_data_dict = dict()
        return output

    @staticmethod
    def load_ly_file(path, with_tokens=False):
        """Loads the LilyPond file into a lines list, so that
        it can be indexed as ``ly_data[line][column]``"""
        with open(path) as hdl:
            lines = [LilyPondLinkPitchParser.clean_line(l)
                     for l in hdl]
        if with_tokens:
            tokens = [l.split() for l in lines]
            return lines, tokens
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
    def check_tie_before_location(line, col, ly_data):
        """Checks whether there is a tie (``~``) in the preceding
        token. Assumes ``col`` points at the beginning of a token."""
        l = ly_data[line]
        _debugprint = 'Looking for tie: line={0}, col={1}\n\tdata: {2}' \
                      ''.format(line, col, l)

        # if l[col-1] not in string.whitespace:
        if col == 0:
            process_prev_line = True
        else:
            ll = l[:col]
            ll_tokens = ll.strip().split()

            _debugprint = 'Looking for tie: line={0}, col={1}\n\tdata: {2}' \
                          '\tll_tokens: {3}'.format(line, col, l, ll_tokens)

            if LilyPondLinkPitchParser.TIE_CHAR in ll:
                logging.debug('--------------------------------')
                logging.debug('There is a tie towards the left!')
                logging.debug(_debugprint)

            # if len(ll_tokens) == 0:
            if not LilyPondLinkPitchParser._find_location_of_last_note(line,
                                                                       ly_data,
                                                                       max_col=col):
                process_prev_line = True
                # logging.debug(_debugprint)
                # logging.debug('Looking at prev. line')
            elif LilyPondLinkPitchParser.TIE_CHAR in ll_tokens[-1]:
                logging.debug('--------------------------------')
                logging.debug(_debugprint)
                logging.debug('Found tie in ll_token!')
                return True
            else:
                return False

        if process_prev_line:
            logging.debug('========================')
            logging.debug('Line {0}: Processing prev. lines'.format(line))
            line -= 1
            col = LilyPondLinkPitchParser._find_location_of_last_note(line, ly_data)
            while not col:
                logging.debug('___________')
                logging.debug('Line {0}: no notes: data {1}'.format(line, ly_data[line]))
                if line == 0:
                    return False
                line -= 1
                col = LilyPondLinkPitchParser._find_location_of_last_note(line, ly_data)

            logging.debug('-------------------------')
            logging.debug(_debugprint)
            logging.debug('Got prev. line {0}, col {1}, data: {2}'.format(line, col, ly_data[line]))

            if LilyPondLinkPitchParser.TIE_CHAR in ly_data[line]:
                logging.debug('previous line {0} has tie char!'.format(line))
                logging.debug('\t\tcol: {0}, tie char position: {1}'
                              ''.format(col, ly_data[line].index(LilyPondLinkPitchParser.TIE_CHAR)))

            if LilyPondLinkPitchParser.TIE_CHAR in ly_data[line][col:]:
                logging.debug(_debugprint)
                logging.debug('Looking at prev. line, found tie! Data: {0}'.format(ly_data[line][col:]))
                return True
            else:
                return False

    @staticmethod
    def ly_line_has_notes(line):
        """Checks whether the given line contains something that can be parsed
        as a note."""
        tokens = line.split()
        has_notes = False
        for t in reversed(tokens):
            try:
                LilyPondLinkPitchParser.ly_token_to_midi_pitch(t)
            except SheetManagerLyParsingError:
                continue
            has_notes = True
            break
        return has_notes

    @staticmethod
    def _find_location_of_last_note(line, ly_data, max_col=None):
        """Tries to find the column at which the rightmost token
        parseable as a note on the given ``line`` of ``ly_data``
        starts. If no such token is found, returns None.
        """
        l = ly_data[line]
        if max_col is not None:
            l = l[:max_col]

        _forward_whitespace_position = len(l)
        _in_token = False
        for i in reversed(range(len(l))):
            if l[i] in string.whitespace:
                if not _in_token:
                    _forward_whitespace_position = i
                    continue

                # Try token
                t = l[i+1:_forward_whitespace_position]
                _in_token = False
                _forward_whitespace_position = i  # single-space
                if LilyPondLinkPitchParser.ly_token_is_note(t):
                    return i + 1

            elif not _in_token:
                _in_token = True

        return None

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
    def ly_token_is_note(ly_token):
        """Checks whether the given token can be parsed as a LilyPond note."""
        try:
            LilyPondLinkPitchParser.ly_token_to_midi_pitch(ly_token)
            return True
        except SheetManagerLyParsingError:
            logging.debug('----- token {0} is not a note!'.format(ly_token))
            return False

    @staticmethod
    def ly_token_to_midi_pitch(ly_token):
        """Converts the LilyPond token into the corresponding MIDI pitch
        code. Assumes the token encodes pitch absolutely."""
        try:
            note = abjad.Note(ly_token)
            wp = note.written_pitch
            midi_code = wp.number + 60
            return midi_code
        except Exception as e:
            # Try stripping away stuff more strictly
            ly_token_strict = ly_token.replace('~', '')
            if ly_token_strict != ly_token:
                try:
                    return LilyPondLinkPitchParser.ly_token_to_midi_pitch(ly_token_strict)
                except Exception as e:
                    raise SheetManagerLyParsingError(e.message)
            else:
                raise SheetManagerLyParsingError(e.message)

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

    :returns: A list of ``(objid, note_idx)`` tuples,
        where the ``objid`` points to the corresponding MuNG object, and
        ``note_idx`` is the MIDI-derived note object.

        Note that (a) not all MIDI matrix onsets have a corresponding visual
        object, (b) not all noteheads have a corresponding onset (ties!).
    """
    ordered_mungo_cols = score.get_ordered_notes(filter_tied=False,
                                                 reverse_columns=True,
                                                 return_columns=True)

    #  - Unroll MIDI matrix (onsets left to right, simultaneous pitches top-down).
    note_events = performance.load_note_events()

    aln = align_mungos_and_note_events_dtw(ordered_mungo_cols, note_events)

    # aln = align_mungos_and_note_events_munkres(ordered_mungos, note_events)
    mdict = dict()
    for m_col in ordered_mungo_cols:
        for m in m_col:
            mdict[m.objid] =  m

    for m_objid, e_idx in aln:
        m = mdict[m_objid]
        e = note_events[e_idx]
        onset_frame = notes_to_onsets([e], dt=1.0 / FPS)
        m.data['{0}_onset_seconds'.format(performance.name)] = e[0]
        m.data['{0}_onset_frame'.format(performance.name)] = int(onset_frame)

    return aln


def align_mungos_and_note_events_dtw(ordered_mungo_columns, events):
    """Align the ordered MuNG columns to the given events using per-frame
    DTW. A column of MuNG objects is one frame in the note space, a set
    of events with the same onset is a frame in the event space.

    The distance function used is "pitch dice": size of intersection between
    the pitch sets over the union of the pitch sets.

    (It could be weighted by the sizes of the columns.)

    Within a column, pitch assignment can be done optimally again by DTW
    on the pitch hinge loss.

    Using DTW guarantees that the alignment is consistent with
    the flow of time, provided that columns are never transposed.
    """
    n_mcols = len(ordered_mungo_columns)
    n_events = len(events)

    m_pitch_sets = []
    for col in ordered_mungo_columns:
        _pitches = []
        for m in col:
            p = m.data['midi_pitch_code']
            try:
                p = int(p)
            except Exception as e:
                print('Note with no pitch got into alignment???\n{0}'
                      ''.format(m))
                raise(e)
            _pitches.append(p)
        _pitches = set(_pitches)
        m_pitch_sets.append(_pitches)

    # m_pitch_sets = [set([int(m.data['midi_pitch_code']) for m in col])
    #                 for col in ordered_mungo_columns]

    # Reverse mapping to event idxs, which we need after
    # grouping events into simultaneity sets
    _e2idx = {tuple(e): i for i, e in enumerate(events)}

    event_simultaneities = collections.defaultdict(list)
    event_simultaneity_idxs = collections.defaultdict(list)
    for e_idx, e in enumerate(events):
        event_simultaneities[e[0]].append(e)
        event_simultaneity_idxs[e[0]].append(e_idx)
    n_ecols = len(event_simultaneities)

    onsets = sorted(event_simultaneities.keys())
    ordered_event_columns = [event_simultaneities[onset] for onset
                             in sorted(event_simultaneities.keys())]
    e_pitch_sets = [set([int(e[1]) for e in e_sim])
                    for e_sim in ordered_event_columns]

    from dtw import dtw
    logging.info('Running DTW: total matrix size: {0}x{1}'
                 ''.format(len(m_pitch_sets), len(e_pitch_sets)))
    dist, cost, acc, path = dtw(m_pitch_sets, e_pitch_sets,
                                dist=lambda x, y:
                                1 - len(x.intersection(y)) / float(len(x.union(y)))
                                )

    # Align the pitch sets against each other.
    # If the DTW path goes only down, multiple MuNG columns correspond to
    # a single onset. This is perfectly OK: e.g., in chords split into voices
    # or with back-to-back noteheads, or if a tied note was accidentally
    # not filtered out.
    # If the DTW path goes only right, multiple onset columns correspond to
    # a single MuNG column. This happens when there are more notes in the MIDI
    # than wirtten, which should be pretty rare, except perhaps if the MIDI
    # rendering engine follows arpeggio instruction.
    # In both cases, we group the MuNG columns / event columns together
    # and assign the groups to each other based on pitch & a single pitch
    # ordering (we do not differentiate between the grouped columns' "onsets")
    logging.info('Processing alignment results. Path length: {0}'.format(len(path[0])))
    # print('Path: {0}'.format(path))
    aln = []
    _i_prev = path[0][0]
    _j_prev = path[1][0]
    _current_m_group = ordered_mungo_columns[_i_prev]
    _current_e_group = ordered_event_columns[_j_prev]
    # ...appending to the paths makes sure to dump
    for i, j in zip(list(path[0][1:]) + [path[0][-1] + 1],
                    list(path[1][1:]) + [path[1][-1] + 1]):

        # print('Positions: {0}, {1}'.format(i, j))

        # If groups are finished:
        if (i != _i_prev) and (j != _j_prev):
            # print('Next grouping: positions {0}, {1} -- groups:'
            #       ' {2}, {3}'.format(i, j, _current_m_group, _current_e_group))
            # Align items in groups
            sorted_m_group = sorted(_current_m_group,
                                    key=lambda _m: _m.data['midi_pitch_code'])
            sorted_e_group = sorted(_current_e_group,
                                    key=lambda _e: _e[1])

            _, _, _, c_path = dtw(sorted_m_group, sorted_e_group,
                                  dist=lambda x, y: x.data['midi_pitch_code'] != int(y[1]))

            if len(sorted_m_group) > 2:
                _frames = [int(numpy.ceil(_e[0] / 0.05)) for _e in sorted_e_group]
                logging.debug('Matched column pitches, frame {2}:'
                              '\n\tMuNG:   {0}'
                              '\n\tEvents: {1}'.format(
                    [_m.data['midi_pitch_code'] for _m in sorted_m_group],
                    [_e[1] for _e in sorted_e_group],
                    _frames[0]
                ))
                logging.debug('\tAln path: {0}'.format(c_path))

            for c_i, c_j in zip(c_path[0], c_path[1]):
                # Only align MuNG object and event if their pitches match!
                # This is what makes the MuNG <--> MIDI relationship reliable.
                if sorted_m_group[int(c_i)].data['midi_pitch_code'] \
                        != sorted_e_group[int(c_j)][1]:
                    continue
                aln.append((sorted_m_group[int(c_i)].objid,
                            _e2idx[tuple( sorted_e_group[int(c_j)] )]
                            ))

            # Clear groups for next
            _current_e_group = []
            _current_m_group = []

        if (i != _i_prev) and (i <= path[0][-1]):
            m_col = ordered_mungo_columns[i]
            _current_m_group.extend(m_col)

        if (j != _j_prev) and (j <= path[1][-1]):
            e_col = ordered_event_columns[j]
            _current_e_group.extend(e_col)

        _i_prev = i
        _j_prev = j

    logging.info('Alignment done, total pairs: {0}'.format(len(aln)))

    return aln
    #
    # simultaneity_aln = [(ordered_mungo_columns[i], ordered_event_columns[j])
    #                     for i, j in zip(path[0], path[1])]
    #
    # # Align pitches within the pitch set. The alignment is supposed
    # # to consist of tuples (m.objid, event_idx). This is where the
    # # reverse dict _e2idx is needed.
    # aln = []
    # for m_col, e_col in simultaneity_aln:
    #     sorted_m_col = sorted(m_col, key=lambda _m: _m.data['midi_pitch_code'])
    #     sorted_e_col = sorted(e_col, key=lambda _e: _e[1])
    #     _, _, _, c_path = dtw(sorted_m_col, sorted_e_col,
    #                           dist=lambda x, y: x.data['midi_pitch_code'] == int(y[1]))
    #     for c_i, c_j in zip(c_path[0], c_path[1]):
    #         print('C_i, c_j = {0}'.format((c_i, c_j)))
    #         if m_col[int(c_i)].data['midi_pitch_code'] != e_col[int(c_j)][1]:
    #             continue
    #         aln.append((m_col[int(c_i)].objid, _e2idx[tuple(e_col[c_j])]))
    #
    # return aln


def align_mungos_and_note_events_munkres(ordered_mungos, note_events, _n_debugplots=10):
    #  - Assign to each MuNG object the MIDI note properties
    if len(note_events) != len(ordered_mungos):
        print('Number of note events and pitched MuNG objects does not'
              ' match: {0} note events, {1} MuNG objs.'
              ''.format(len(note_events), len(ordered_mungos)))

    output = []

    event_idx = 0
    mung_idx = 0
    while (event_idx < len(note_events)) \
        and (mung_idx < len(ordered_mungos)):

        m = ordered_mungos[mung_idx]
        e = note_events[event_idx]

        pitch_m = int(m.data['midi_pitch_code'])
        pitch_e = int(e[1])

        if pitch_m != pitch_e:
            print('Pitch of MuNG object {0} and corresponding MIDI note {1}'
                  ' with onset {2} does not'
                  ' match: MuNG {3}, event {4}'.format(m.objid, event_idx,
                                                       e[0],
                                                       pitch_m, pitch_e))
            print('Falling back on munkres.')
            # Apply munkres going forward.

            N = 10

            # The MuNG objects must be part of the same system!
            m_snippet = ordered_mungos[mung_idx:mung_idx + N]
            _m_snippet_idxs = {m.objid: i+1 for i, m in enumerate(m_snippet)}
            # We need the snippet idxs to correctly assign event idxs
            # in case Munkres skips an event, and to correctly move the
            # e_idx and m_idx.

            e_snippet = note_events[event_idx:event_idx + N]
            _e_snippet_idxs = {tuple(e): i+1 for i, e in enumerate(e_snippet)}

            snippet_aln = munkres_align_snippet(m_snippet, e_snippet,
                                                _debugplot=(_n_debugplots > 0))
            _n_debugplots -= 1
            # Alignment still may contain pairs with mismatched pitches.

            # Incorporating this back into the overall alignment:
            # not all the MuNG objects were necessarily aligned.
            # There may have been extra note events, which means that
            # we did not give the MuNGs enough note events to align to.
            # On the other hand, they may have been legitimate FPs.

            # One solution is to take only the first K < N elements
            # from the alignment.

            K = 5

            m_idx_shift = 0
            e_idx_shift = 0
            for am, ae in snippet_aln[:K]:
                if am.data['midi_pitch_code'] != ae[1]:
                    print('Munkres did not work on m.objid={0}/{1}, e={2:.2f}/{3}'
                          ''.format(am.objid, am.data['midi_pitch_code'], ae[0], ae[1]))
                    continue
                onset_frame = notes_to_onsets([ae], dt=1.0 / FPS)
                #am.data['{0}_onset_seconds'.format(performance.name)] = ae[0]
                #am.data['{0}_onset_frame'.format(performance.name)] = int(onset_frame)
                print('Munkres WORKED on m.objid={0}/{1}, e={2:.2f}/{3}'
                      ''.format(am.objid, am.data['midi_pitch_code'], ae[0], ae[1]))

                output.append((am.objid, event_idx + _e_snippet_idxs[tuple(ae)] - 1))

                # Check how much move the index.
                m_idx_shift = max(m_idx_shift, _m_snippet_idxs[am.objid])
                e_idx_shift = max(e_idx_shift, _e_snippet_idxs[tuple(ae)])

            print('Shifting out of {2} by: m={0}, e={1}'
                  ''.format(m_idx_shift, e_idx_shift, K))

            if (m_idx_shift == 0) and (e_idx_shift == 0):
                print('Got stuck: nothing gets assigned to each other at index'
                      ' positions m={0}, e={1}'.format(mung_idx, event_idx))
                return output

            mung_idx += m_idx_shift
            event_idx += e_idx_shift

        else:
            print('Pitch of MuNG object {0} and corresponding MIDI note {1}'
                  ' with onset {2} matches:'
                  'MuNG {3}, event {4}'.format(m.objid, event_idx,
                                                       e[0],
                                                       pitch_m, pitch_e))

            onset_frame = notes_to_onsets([e], dt=1.0 / FPS)
            #m.data['{0}_onset_seconds'.format(performance.name)] = e[0]
            #m.data['{0}_onset_frame'.format(performance.name)] = int(onset_frame)

            output.append((m.objid, event_idx))

            event_idx += 1
            mung_idx += 1

    return output


def munkres_align_snippet(mungos, events, _debugplot=False):
    """Aligns the given MuNG objects and mm.note events using the munkres
    algorithm."""
    import munkres

    # prepare cost structures

    pitch_mismatch_cost = 3

    # events = sorted(events, key=lambda x: (x[0], x[1] * -1))

    event_start = min([e[0] for e in events])
    event_end = max([e[0] + e[2] for e in events])
    event_span = float(event_end - event_start)
    event_proportional_starts = [(e[0] - event_start) / event_span
                                 for e in events]

    # This does not work if events cross systems...
    # mungos = sorted(mungos, key=lambda x: (x.left, x.top))

    mung_start = min([m.left for m in mungos])
    mung_end = max([m.right for m in mungos])
    mung_span = float(mung_end - mung_start)
    mungo_proportional_starts = [(m.left - mung_start) / mung_span
                                 for m in mungos]

    aln = None

    no_valid_assignment_found = True
    disallowed_pairs = []
    MAX_MUNKRES_ATTEMPTS = 8
    _n_munkres_attempts = 0
    while no_valid_assignment_found:
        _n_munkres_attempts += 1

        # Compute distances matrix
        D = numpy.zeros((len(mungos), len(events)), dtype=numpy.float)

        # ...add a "sink" row and column for un-alignable things?

        for i, m in enumerate(mungos):
            for j, e in enumerate(events):

                cost = 0.0

                m_position = mungo_proportional_starts[i]
                e_position = event_proportional_starts[j]
                prop_span = numpy.abs(m_position - e_position)
                cost += prop_span

                pitch_m = m.data['midi_pitch_code']
                pitch_e = int(e[1])
                if pitch_m != pitch_e:
                    # print('m-{0}: {1}, e-{2}: {3}'.format(m.objid, pitch_m, j, e[1]))
                    cost += pitch_mismatch_cost

                D[i, j] = cost

                if (m.objid, tuple(e)) in disallowed_pairs:
                    D[i, j] = numpy.inf

        # Add disallowed entries:

        # init minimum assignment algorithm
        mkr = munkres.Munkres()

        print('Computing munkres...')
        if D.shape[0] <= D.shape[1]:
            assignment = numpy.asarray(sorted(mkr.compute(D.copy())))
        else:
            assignment = numpy.asarray(sorted(mkr.compute(D.T.copy())))
            assignment = assignment[:, ::-1]
        print('Done!')

        if _debugplot:
            import matplotlib
            matplotlib.use('Qt4Agg')
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(121)
            plt.imshow(D.T)
            plt.xticks(range(len(mungos)), [(m.objid, m.data['midi_pitch_code']) for m in mungos],
                       rotation='vertical')
            plt.yticks(range(len(events)), ['{0:.2f}: {1}'.format(e[0], int(e[1]))
                                            for e in events])
            plt.xlabel('MuNG objects')
            plt.ylabel('MIDI events')

            #######
            # Plot alignment results
            aln_map = numpy.zeros(D.shape)
            for i, j in assignment:
                aln_map[i, j] = 1
                if mungos[i].data['midi_pitch_code'] != events[j][1]:
                    aln_map[i, j] = 0.5
            plt.subplot(122)
            plt.imshow(aln_map.T)
            plt.xticks(range(len(mungos)),
                       [(m.objid, m.data['midi_pitch_code']) for m in mungos],
                       rotation='vertical')
            plt.yticks(range(len(events)), ['{0:.2f}: {1}'.format(e[0], int(e[1]))
                                            for e in events])
            plt.xlabel('MuNG objects')
            plt.ylabel('MIDI events')
            plt.show()

        ########
        # Produce alignment & check for conflicts
        aln = [(mungos[i], events[j]) for i, j in assignment]

        conflict_pair = find_conflict_in_alignment(aln, mungos, events, D)

        if conflict_pair:
            cm, ce = conflict_pair
            _conflict_pair_hash = (cm.objid, tuple(ce))
            print('Attempt {0}: found conflict pair: {1}'.format(_n_munkres_attempts,
                                                                 _conflict_pair_hash))
            disallowed_pairs.append(_conflict_pair_hash)
            print('Disallowed pairs: {0}'
                  ''.format(disallowed_pairs))
            if _n_munkres_attempts > MAX_MUNKRES_ATTEMPTS:
                print('Giving up: over 10 munkres attempts!')
                break
        else:
            no_valid_assignment_found = False

            print('Munkres: alignment found in {0} attempts'
                  ''.format(_n_munkres_attempts))

    return aln


def find_conflict_in_alignment(aln, mungos, events,
                               cost_matrix,
                               columns_strict=True):
    """Check if alignment is valid. If it is not, forbid
    the invalid cells next time.
    Alignment is invalid if the ordering of MuNGs
    and events is inconsistent.
    MuNG columns are equivalent with respect to onsets:
    we don't care about differences within a column,
    but if we move to the next column, the min. event onset must
    not go *back*."""
    mung_columns = group_mungos_by_column(mungos)
    mung_lefts_per_column = {}
    for l, col in mung_columns.items():
        for m in col:
            mung_lefts_per_column[m.objid] = l

    # This is the output variable: a pair ``(m, e)``.
    conflict_pair = None

    _event_list = [tuple(e) for e in events]
    _alndict = {m.objid: e for m, e in aln}
    _onset_dict = {m.objid: e[0] for m, e in aln}
    _onset_reverse_dict = collections.defaultdict(list)
    for m, e in aln:
        _onset_reverse_dict[e[1]].append(m)

    # Strict "onset column" handling:
    # All notes with the same onset should be in the same column.
    for onset in _onset_reverse_dict:
        onset_mungos = _onset_reverse_dict[onset]
        onset_cols = [mung_lefts_per_column[m.objid] for m in onset_mungos]

        if len(set(onset_cols)) <= 1:
            continue

        # If there are notes from multiple columns aligned to one onset:
        #  - Disallow the highest-cost assignment on the onset.
        _max_cost, _max_cost_pair = -numpy.inf, None
        for m in onset_mungos:
            e = _alndict[m.objid]
            i, j = mungos.index(m), _event_list.index(tuple(e))
            cost = cost_matrix[i, j]
            if cost > _max_cost:
                _max_cost = cost
                _max_cost_pair = m, e

        conflict_pair = _max_cost_pair
        print('Found conflict: same ONSET, multiple COLUMNS!'
              'Maximum cost pair in conflict column:'
              ' m: {0}/{1}, e: {2:.3f}/{3}'.format(_max_cost_pair[0].objid,
                                               _max_cost_pair[0].data['midi_pitch_code'],
                                               _max_cost_pair[1][0],
                                               _max_cost_pair[1][1]))

    if conflict_pair is not None:
        return conflict_pair

    # Strict MuNG column handling:
    # All notes within a column should have the same onset.
    for l, col in mung_columns.items():
        column_onset_counts = collections.defaultdict(int)
        for m in col:
            column_onset_counts[_alndict[m.objid][0]] += 1

        if len(column_onset_counts) == 1:
            continue

        # If there are multiple onsets within a column:
        #  - Disallow the highest-cost assignment in the column.
        _max_cost = -numpy.inf
        _max_cost_pair = None
        for m in col:
            e = _alndict[m.objid]
            i, j = mungos.index(m), _event_list.index(tuple(e))
            cost = cost_matrix[i, j]
            if cost > _max_cost:
                _max_cost = cost
                _max_cost_pair = m, e

        conflict_pair = _max_cost_pair
        print('Found conflict: same COLUMN, multiple ONSETS!'
              'Maximum cost pair in conflict column:'
              ' m: {0}/{1}, e: {2:.3f}/{3}'.format(_max_cost_pair[0].objid,
                                               _max_cost_pair[0].data['midi_pitch_code'],
                                               _max_cost_pair[1][0],
                                               _max_cost_pair[1][1]))

    if conflict_pair is not None:
        return conflict_pair

    # Sort MuNGs by onset and find conflicts w.r.t their column left
    # (transposition)
    conflict_mungs = None

    for _i, (m1, e1) in enumerate(sorted(aln, key=lambda kv: kv[1][0])):
        for m2, e2 in sorted(aln, key=lambda kv: kv[1][0])[_i:]:
            # If the *later* note is written *left*:
            if mung_lefts_per_column[m2.objid] < mung_lefts_per_column[m1.objid]:
                # If their pitches match (otherwise this will get removed anyway):
                if m1.data['midi_pitch_code'] == m2.data['midi_pitch_code']:
                    conflict_mungs = (m1, m2)
                break
        if conflict_mungs is not None:
            break

    if conflict_mungs is not None:
        # We could clear the disallowed pairs here,
        # but let's keep it greedy for now.
        m1, m2 = conflict_mungs
        # The MuNG/event pair with the higher cost is marked as forbidden.
        e1 = _alndict[m1.objid]
        e2 = _alndict[m2.objid]

        i1, j1 = mungos.index(m1), _event_list.index(tuple(e1))
        i2, j2 = mungos.index(m2), _event_list.index(tuple(e2))
        cost1 = cost_matrix[i1, j1]
        cost2 = cost_matrix[i2, j2]
        if cost1 > cost2:
            conflict_pair = m1, events[j1]
        else:
            conflict_pair = m2, events[j2]

        print('Found conflict: TRANSPOSITION!'
              'Maximum cost pair in conflict column(s):'
              ' m: {0}/{1}, e: {2:.3f}/{3}'.format(conflict_pair[0].objid,
                                               conflict_pair[0].data['midi_pitch_code'],
                                               conflict_pair[1][0],
                                               conflict_pair[1][1]))
    return conflict_pair


def group_mungos_by_system(page_mungos, score_img=None, page_num=None,
                           MIN_PEAK_WIDTH=5, NONWHITE_COLUMNS_TOLERANCE=0.001):
    """Groups the MuNG objects on a page into systems. Assumes
    piano music: there is a system break whenever a pitch that
    overlaps horizontally and is lower on the page is higher
    than the previous pitch.

    Only takes into account MuNG objects with
    the ``midi_pitch_code`` in their ``data`` dict.

    This method assumes no systems have been detected.

    :returns: ``(system_bboxes, system_mungos)`` where ``system_bboxes``
        are ``(top, left, bottom, right)`` tuples denoting the suggested
        system bounding boxes, and ``system_mungos`` is a list of MuNG
        objects
    """
    if len(page_mungos) < 2:
        logging.warning('Grouping MuNG objects by system'
                        ' called with only {0} objects!'
                        ''.format(len(page_mungos)))
        return [page_mungos]

    page_mungos = [m for m in page_mungos
                   if 'midi_pitch_code' in m.data]

    sorted_mungo_columns = group_mungos_by_column(page_mungos)

    logging.debug('Total MuNG object columns: {0}'
                  ''.format(len(sorted_mungo_columns)))
    logging.debug('MuNG column lengths: {0}'
                  ''.format(numpy.asarray([len(col)
                                           for col in sorted_mungo_columns.values()])))

    dividers = find_column_divider_regions(sorted_mungo_columns)

    logging.debug('Dividers: {0}'.format(dividers))

    # Now, we take the horizontal projection of the divider regions
    canvas_height = max([m.bottom for m in page_mungos])
    canvas_width = max([m.right for m in page_mungos])
    canvas_size = canvas_height + 5, \
                  canvas_width + 5
    canvas = numpy.zeros(canvas_size, dtype='uint8')
    for t, l, b, r in dividers:
        canvas[t:b, l:r] += 1

    dividers_hproj = canvas.sum(axis=1)

    allowed_peaks_hproj = numpy.ones_like(dividers_hproj)

    if score_img is not None:
        # Get image horizontal projections, restrict
        # peaks to areas where the image hproj. is maximal
        import cv2
        if score_img.ndim == 3:
            img = numpy.average(score_img, axis=0)
        else:
            img = score_img
        blur = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=2.0, sigmaY=2.0)
        _, img_binarized = cv2.threshold(blur, 0, blur.max(),
                                         cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_hproj = img_binarized.sum(axis=1)[:dividers_hproj.shape[0]]
        img_hproj_norm = img_hproj / img_binarized.max()
        max_img_hproj = max(img_hproj_norm)
        # NONWHITE_COLUMNS_TOLERANCE = 0.02
        print('Allowed peaks threshold: {0}, with maximum {1}'
              ''.format(max_img_hproj * (1 - NONWHITE_COLUMNS_TOLERANCE), max_img_hproj))
        allowed_peaks_hproj = img_hproj_norm >= (max_img_hproj * (1 - NONWHITE_COLUMNS_TOLERANCE))


    ### DEBUG
    # or very close to maximal.
    if (score_img is not None) and (page_num is not None):
        import matplotlib
        matplotlib.use('Qt4Agg')
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(score_img[:canvas_height, :canvas_width], cmap='gray')
        plt.imshow(canvas[:canvas_height, :canvas_width], alpha=0.3)

        plt.plot(dividers_hproj, numpy.arange(dividers_hproj.shape[0]))
        if allowed_peaks_hproj is not None:
            plt.plot(allowed_peaks_hproj * 100, numpy.arange(dividers_hproj.shape[0]))
        plt.show()

    # Now we find local peaks (or peak areas) of the projections.
    # - what is a peak? higher than previous, higher then next
    #   unequal
    peak_starts = []
    peak_ends = []

    peak_start_candidate = 0
    ascending = False
    for i in range(1, dividers_hproj.shape[0] - 1):
        # If current is higher than previous:
        #  - previous cannot be a peak.
        if dividers_hproj[i] > dividers_hproj[i - 1]:
            peak_start_candidate = i
            ascending = True
        # If current is higher than next:
        #  - if we are in an ascending stage: ascent ends, found peak
        if dividers_hproj[i] > dividers_hproj[i + 1]:
            if not ascending:
                continue
            peak_starts.append(peak_start_candidate)
            peak_ends.append(i)
            ascending = False

    logging.debug('Peaks: {0}'.format(zip(peak_starts, peak_ends)))

    if len(peak_starts) == 0:
        # no peaks: only one system => all MuNG objects
        # go into one system
        t, l, b, r = cropobjects_merge_bbox(page_mungos)
        return [(t, l, b, r)], [page_mungos]


    # Filter out very sharp peaks
    peak_starts, peak_ends = map(list, zip(*[(s, e)
                                             for s, e in zip(peak_starts, peak_ends)
                                   if (e - s) > MIN_PEAK_WIDTH]))

    # Factor in the allowed peaks projection
    peak_proj = numpy.zeros_like(allowed_peaks_hproj)
    for s, e in zip(peak_starts, peak_ends):
        peak_proj[s:e] = 1
    logging.debug('Peak projection: {0} nonzero rows'.format(peak_proj.sum()))
    peak_proj *= allowed_peaks_hproj
    logging.debug('After applying allowed peaks: {0} nonzero rows'.format(peak_proj.sum()))
    logging.debug('Allowed peak locations: {0}'.format(allowed_peaks_hproj.sum()))
    peaks = greater_than_zero_intervals(peak_proj)
    peak_starts, peak_ends = map(list, zip(*peaks))

    # Use peaks as separators between system regions.
    system_regions = []
    for s, e in zip([0] + peak_ends, peak_starts + [canvas_height]):
        region = (s+1, 1, e, canvas_width)
        system_regions.append(region)

    logging.debug('System regions:\n{0}'.format([(t, b) for t, l, b, r in system_regions]))

    system_mungos = group_mungos_by_region(page_mungos, system_regions)

    # Crop system boundaries based on mungos
    # (includes filtering out systems that have no objects)
    cropped_system_boundaries = []
    cropped_system_mungos = []
    for mungos in system_mungos:
        if len(mungos) == 0:
            continue
        t, l, b, r = cropobjects_merge_bbox(mungos)
        cropped_system_boundaries.append((t, l, b, r))
        cropped_system_mungos.append(mungos)

    # Merge vertically overlapping system regions
    sorted_system_boundaries = sorted(cropped_system_boundaries, key=lambda x: x[0])
    merge_sets = []
    current_merge_set = [0]
    for i in range(len(sorted_system_boundaries[:-1])):
        t, l, b, r = sorted_system_boundaries[i]
        nt, nl, nb, nr = sorted_system_boundaries[i+1]
        if nt <= b:
            current_merge_set.append(i + 1)
        else:
            merge_sets.append(copy.deepcopy(current_merge_set))
            current_merge_set = [i+1]
    merge_sets.append(copy.deepcopy(current_merge_set))

    logging.debug('Merge sets: {0}'.format(merge_sets))

    merged_system_boundaries = []
    for ms in merge_sets:
        regions = [sorted_system_boundaries[i] for i in ms]
        if len(regions) == 1:
            logging.debug('No overlap for merge set {0}, just adding it'
                          ''.format(regions))
            merged_system_boundaries.append(regions[0])
            continue
        mt = min([r[0] for r in regions])
        ml = min([r[1] for r in regions])
        mb = max([r[2] for r in regions])
        mr = max([r[3] for r in regions])
        merged_system_boundaries.append((mt, ml, mb, mr))
        logging.debug('Merging overlapping systems: ms {0}, regions {1}, '
                      'boundary: {2}'.format(ms, regions, (mt, ml, mb, mr)))
    merged_system_mungos = group_mungos_by_region(page_mungos,
                                                  merged_system_boundaries)

    return merged_system_boundaries, merged_system_mungos


def group_mungos_by_region(page_mungos, system_regions):
    """Group MuNG objects based on which system they belong to."""
    system_mungos = [[] for _ in system_regions]
    # TODO
    # For each MuNG object, find the closest region
    for i, (t, l, b, r) in enumerate(system_regions):
        for m in page_mungos:
            if m.overlaps((t, l, b, r)):
                system_mungos[i].append(m)

    return system_mungos


def find_column_divider_regions(sorted_mungo_columns):
    """Within each MuNG note column, use the MIDI pitch code data
    attribute to find suspected system breaks."""
    rightmost_per_column = {l: max([m.right
                                    for m in sorted_mungo_columns[l]])
                            for l in sorted_mungo_columns}
    # Now we have the MuNG objects grouped into columns.
    # Next step: find system breaks in each column.
    system_breaks_mungos_per_col = collections.defaultdict(list)
    # Collects the pairs of MuNG objects in each column between
    # which a page break is suspected.
    for l in sorted_mungo_columns:
        m_col = sorted_mungo_columns[l]
        system_breaks_mungos_per_col[l] = []
        if len(m_col) < 2:
            continue
        for m1, m2 in zip(m_col[:-1], m_col[1:]):
            logging.debug('Col {0}: comparing pitches {1}, {2}'
                          ''.format(l, m1.data['midi_pitch_code'],
                                    m2.data['midi_pitch_code']))
            # Noteheads very close togehter in a column..?
            if (m2.top - m1.top) < m1.height:
                continue
            if m1.data['midi_pitch_code'] < m2.data['midi_pitch_code']:
                system_breaks_mungos_per_col[l].append((m1, m2))
    logging.debug('System breaks: {0}'
                  ''.format(pprint.pformat(dict(system_breaks_mungos_per_col))))
    # We can now draw dividing regions where we are certain
    # a page brerak should occur.
    dividers = []
    for l in system_breaks_mungos_per_col:
        r = rightmost_per_column[l]
        for m1, m2 in system_breaks_mungos_per_col[l]:
            t = m1.bottom + 1
            b = m2.top
            dividers.append((t, l, b, r))
    return dividers


def group_mungos_by_system_paths(page_mungos, score_img, page_num=None,
                                 CONNECTIVITY_TOLERANCE=10,
                                 _debugplot=False):
    """Groups the MuNG objects (assumed only notes) by system using separating
    paths. A separating path is a background-only path that connects the left
    and right limit of the region in which the MuNG objects are.

    The grouping algorithm finds all separating paths and uses the union of
    the pixels belonging to these paths as a mask to divide
    the image into system regions. All MuNG objects that fall within the same
    system region are then grouped together.

    Separating paths pixels are pixels of connected components of the (white)
    background of the image region delimited by MuNG that touch both the left
    and right edge of this region.

    :param connectivity_tolerance: How many pixels from the edge can a system
        start. Useful for first systems that are usually shorter, because
        the instrument name is on their left side.
    """

    pt, pl, pb, pr = cropobjects_merge_bbox(page_mungos)
    canvas = score_img[:pb, :pr]

    # This bounding box is too small in pieces with single-measure lines.
    # It needs to include the multi-staff braces at the beginning.
    # ...maybe an option is to project the "empty rows" leftward
    #    and only keep the separating region if at least 10 % of its
    #    rows project leftward without interruption?

    import cv2
    _, canvas_bin = cv2.threshold(canvas, 0, canvas.max(),
                                 cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, labels = cv2.connectedComponents(canvas_bin, connectivity=8)
    left_side_labels = set(labels[:, 0])
    right_side_labels = set(labels[:, -1])
    separating_label_candidates = left_side_labels.intersection(right_side_labels)
    # Filter out background label
    if 0 in separating_label_candidates:
        separating_label_candidates = [l for l in separating_label_candidates
                                       if l != 0]

    separating_labels = separating_label_candidates
    #
    # # Now apply the projection to the left, to filter out single-measure systems
    # _, left_img_bin = cv2.threshold(score_img[pt:pb, :pl], 0, score_img.max(),
    #                                 cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # left_img_bin[left_img_bin != left_img_bin.max()] = 2
    # left_img_bin[left_img_bin > 2] = 0
    # left_img_bin[left_img_bin != 0] = 1
    #
    # left_separating_labels = labels[:, :2]
    # left_sep_regions = regionprops(left_separating_labels)
    #
    # separating_labels = []
    # for lsr in left_sep_regions:
    #     if lsr.label not in separating_label_candidates:
    #         continue
    #
    #     lsr_t, lsr_l, lsr_b, lsr_r = lsr.bbox
    #     # In this crop, sheet background is 0 and non-bg is 1
    #     left_img_crop = left_img_bin[lsr_t:lsr_b]
    #     left_img_hproj = left_img_crop.sum(axis=1)
    #     # Are all rows non-zero?
    #
    #     if len(left_img_hproj.nonzero()[0]) == 0:
    #
    #         if _debugplot:
    #             import matplotlib
    #             matplotlib.use('Qt4Agg')
    #             import matplotlib.pyplot as plt
    #             plt.figure()
    #             plt.title('Discarding sep. region, rows: {0}--{1}'.format(lsr_t, lsr_b))
    #             plt.imshow(left_img_crop, cmap='gray', interpolation='nearest')
    #             plt.plot(left_img_hproj * 5, numpy.arange(left_img_hproj.shape[0]))
    #             plt.show()
    #
    #         continue
    #
    #     separating_labels.append(lsr.label)

    if len(separating_labels) == 0:
        print('No separating path found!')

    # # Find shortest paths for the separating labels
    # for l in separating_labels:
    #     pass

    separated_region_mask = numpy.ones(labels.shape, dtype='uint8')
    for l in separating_labels:
        # Skip background
        if l == 0:
            continue
        separated_region_mask[labels == l] = 0

    _, system_labels = cv2.connectedComponents(separated_region_mask,
                                               connectivity=8)

    system_regions = regionprops(system_labels)
    separating_system_labels = []
    _c_height, _c_width = system_labels.shape
    for sr in system_regions:
        _is_system = False
        srt, srl, srb, srr = sr.bbox
        if (srl < CONNECTIVITY_TOLERANCE) \
                and (srr > (_c_width - CONNECTIVITY_TOLERANCE)):
            _is_system = True

        if ((srb - srt) > (_c_height * 0.08)) \
            and ((srr - srl) > _c_width * 0.5) \
                and (sr.extent > 0.5) \
                and (sr.area > (0.05 * _c_height * _c_width)):
            _is_system = True

        if _is_system:
            separating_system_labels.append(sr.label)

    # Only retain as systems labels that connect from left to right.
    # CONNECTIVITY_TOLERANCE = 10
    # left_side_system_labels = set(system_labels[:, 0 + CONNECTIVITY_TOLERANCE])
    # right_side_system_labels = set(system_labels[:, -1 - CONNECTIVITY_TOLERANCE])
    # separating_system_labels = left_side_system_labels.intersection(
    #                                             right_side_system_labels)
    # separating_system_labels = [l for l in separating_system_labels if l != 0]
    separating_syslabel_image = numpy.zeros(system_labels.shape, dtype='uint8')
    for l in separating_system_labels:
        separating_syslabel_image[system_labels == l] = l

    # Those that do not should be assigned to the closest system.
    non_separating_syslabel_image = system_labels * 1
    for l in separating_system_labels:
        non_separating_syslabel_image[non_separating_syslabel_image == l] = 0

    non_separating_syslabel_assignment = dict()
    regions = regionprops(non_separating_syslabel_image)
    for r in regions:
        rt, rl, rb, rr = r.bbox
        dvert = 1
        while ((rt - dvert) >= 0) or ((dvert + rb) < pb):
            # We might hit the iteration limit in the top and bottom system
            # on the page.
            if (rt - dvert) >= 0:
                upper_slice = separating_syslabel_image[rt - dvert, rl:rr]
                upper_nnz = list(upper_slice[upper_slice.nonzero()])
                if len(upper_nnz) > 0:
                    # Assign to the found region & break
                    separating_system_label = min(upper_nnz)
                    non_separating_syslabel_assignment[r.label] = separating_system_label
                    break

            if (rb + dvert) < pb:
                lower_slice = separating_syslabel_image[rb + dvert, rl:rr]
                lower_nnz = list(lower_slice[lower_slice.nonzero()])
                if len(lower_nnz) > 0:
                    separating_system_label = min(lower_nnz)
                    non_separating_syslabel_assignment[r.label] = separating_system_label
                    break

            dvert += 1

    for nonsep_l, sep_l in non_separating_syslabel_assignment.items():
        system_labels[system_labels == nonsep_l] = sep_l

    # Now group the MuNG objects
    system_groups = collections.defaultdict(list)
    for m in page_mungos:
        # DON'T translate MuNG-O bounding box top to canvas now:
        ct, cl, cb, cr = m.top, m.left, m.bottom, m.right
        # ct, cl, cb, cr = m.bounding_box
        # Can use the MuNG-O mask to make the intersection more accurate,
        # but this doesn't matter much for noteheads at 835 px page width.
        # (It may matter down the road for other symbols.)
        m_labels = set(list(system_labels[ct:cb, cl:cr].flatten()))
        for l in m_labels:
            if l != 0:
                system_groups[l].append(m)

    # Remove all background notes
    if 0 in system_groups:
        del system_groups[0]

    # Extract just the groups themselves (decouple from original labels)
    system_groups = system_groups.values()

    # And create the regions:
    system_bboxes = [cropobjects_merge_bbox(system_group)
                     for system_group in system_groups]
    # Transpose the bboxes w.r.t. original image is not necessary, because
    # the group contains the original MuNG objects with coords w.r.t. page
    # system_bboxes = [(st, sl + pl, sb, sr + pl)
    #                  for st, sl, sb, sr in system_bboxes]
    system_groups, system_bboxes = zip(*[(g, b)
                                         for g, b, in sorted(zip(system_groups,
                                                                 system_bboxes),
                                                             key=lambda kv: kv[1])
                                         if len(g) > 0])

    # Debugging plot:
    if _debugplot:
        import matplotlib
        matplotlib.use('Qt4Agg')
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(canvas_bin, cmap='gray', interpolation='nearest')
        _img_system_labels = system_labels * 1
        _img_system_labels[_img_system_labels != 0] += 10
        plt.imshow(_img_system_labels, interpolation='nearest', alpha=0.6)
        plt.title('Detected system regions, page {0}'.format(page_num))
        plt.show()

    return system_bboxes, system_groups


def build_system_mungos_on_page(system_boundaries, system_mungo_groups,
                                start_objid):
    """Creates the ``staff`` MuNG objects from the given system
    boudnaries. Adds the inlinks/outlinks to the other MuNG object
    groups -- modifies the objects in-place. Assumes each system
    has at least one MuNG in its group.
    """
    system_mungs = []
    _current_objid = start_objid
    for sb, smg in zip(system_boundaries, system_mungo_groups):
        if len(smg) == 0:
            continue
        m = smg[0]
        uid = CropObject.build_uid(m.dataset,
                                   m.doc,
                                   _current_objid)
        t, l, b, r = CropObject.bbox_to_integer_bounds(*sb)
        h, w, = b - t, r - l
        mask = numpy.ones((h, w))
        system_mung = CropObject(objid=_current_objid,
                                 clsname='staff',
                                 top=t, left=l, height=h, width=w,
                                 mask=mask,
                                 uid=uid,
                                 data=dict())
        system_mungs.append(system_mung)

        _current_objid += 1

        for m in smg:
            link_cropobjects(m, system_mung, check_docname=False)

    return system_mungs


def alignment_stats(mungos, events, aln):
    """Compute the hits, misses, and tied misses for the given
    MuNG -- Events alignment.

    (You can call this per-page.)

    The result can help flag a suspicious piece, based on some criteria
    on the stats.
    """
    mdict = {m.objid: m for m in mungos}
    aln_dict = {objid: e_idx for objid, e_idx in aln.items()}
    if isinstance(events, dict):
        event_hits = {e_idx: 0 for e_idx in events}
    else:
        event_hits = {e_idx: 0 for e_idx in range(len(events))}

    n_mungos = len(mungos)
    n_events = len(event_hits)
    n_aln_pairs = len(aln)

    mungos_not_aligned_not_tied = []
    mungos_not_aligned_tied = []
    mungos_aligned_wrong_pitch = []
    mungos_aligned_correct_pitch = []
    mungos_aligned_no_pitch = []
    mungos_not_aligned_no_pitch = []

    events_without_corresponding_mungo = []
    events_with_multiple_mungos = []

    system_mungos = []

    for m in mungos:
        if m.objid not in aln_dict:
            if m.clsname == 'staff':
                system_mungos.append(m)
                continue

            if ('tied' in m.data) and (m.data['tied'] == 1):
                mungos_not_aligned_tied.append(m)
            else:
                mungos_not_aligned_not_tied.append(m)

            if 'midi_pitch_code' not in m.data:
                mungos_not_aligned_no_pitch.append(m)

            continue

        m_pitch = None
        if 'midi_pitch_code' not in m.data:
            mungos_aligned_no_pitch.append(m)
        else:
            m_pitch = m.data['midi_pitch_code']

        e_idx = aln_dict[m.objid]
        event_hits[e_idx] += 1

        event = events[e_idx]
        e_pitch = int(event[1])
        # e_onset_s = event[0]
        # e_onset_frame = notes_to_onsets([event], 1.0 / FPS)[0]

        if m_pitch == e_pitch:
            mungos_aligned_correct_pitch.append(m)
        else:
            mungos_aligned_wrong_pitch.append(m)

    for e_idx, n_hits in event_hits.items():
        if n_hits == 0:
            e = events[e_idx]
            events_without_corresponding_mungo.append(e)
        elif n_hits > 1:
            e = events[e_idx]
            events_with_multiple_mungos.append(e)

    AlnStats = collections.namedtuple('AlnStats',
                                      ['mungos_not_aligned_not_tied',
                                       'mungos_not_aligned_tied',
                                       'mungos_aligned_wrong_pitch',
                                       'mungos_aligned_correct_pitch',
                                       'mungos_aligned_no_pitch',
                                       'mungos_not_aligned_no_pitch',
                                       'events_without_corresponding_mungo',
                                       'events_with_multiple_mungos',
                                       'n_mungos',
                                       'n_events',
                                       'n_aln_pairs',
                                       'system_mungos']
                                      )

    stats = AlnStats(mungos_not_aligned_not_tied=mungos_not_aligned_not_tied,
                     mungos_not_aligned_tied=mungos_not_aligned_tied,
                     mungos_aligned_wrong_pitch=mungos_aligned_wrong_pitch,
                     mungos_aligned_correct_pitch=mungos_aligned_correct_pitch,
                     mungos_aligned_no_pitch=mungos_aligned_no_pitch,
                     mungos_not_aligned_no_pitch=mungos_not_aligned_no_pitch,
                     events_without_corresponding_mungo=events_without_corresponding_mungo,
                     events_with_multiple_mungos=events_with_multiple_mungos,
                     n_mungos=n_mungos,
                     n_events=n_events,
                     n_aln_pairs=n_aln_pairs,
                     system_mungos=system_mungos)

    return stats


def is_aln_problem(stats):
    """Returns a guess whether the given stats point to a problem
    in the alignment."""
    is_problem = False

    n_probable_errors = len(stats.mungos_not_aligned_not_tied)
    n_probable_correct = len(stats.mungos_aligned_correct_pitch)
    p_red = float(n_probable_errors) / float(n_probable_correct)
    if p_red > 0.05:
        is_problem = True

    return is_problem
