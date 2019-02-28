"""This module implements LilyPond file parsing (to the extent that we need:
extracting pitches based on the point-and-click backlinks)."""
from __future__ import print_function

import logging
import string

import abjad

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


class MSMDLyParsingError(Exception):
    pass


def mung_midi_from_ly_links(cropobjects):
    """Adds the ``midi_pitch_code`` data attribute for all CropObjects
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
            except MSMDLyParsingError as e:
                raise MSMDLyParsingError('Token {0} at location {1}'
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
            except MSMDLyParsingError:
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
        except MSMDLyParsingError:
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
                    raise MSMDLyParsingError(e)
            else:
                raise MSMDLyParsingError(e)

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
