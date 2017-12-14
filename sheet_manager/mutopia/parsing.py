"""This module implements utilities for handling Mutopia files."""
from __future__ import print_function, unicode_literals

import codecs
import collections
import logging
import os

import re
import shlex
import subprocess

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################


INCLUDE_TOKEN = '\\include'
VERSION_TOKEN = '\\version'
UNFOLD_REPEATS_TOKEN = '\\unfoldRepeats'
PIANO_LIKE_INSTRUMENTS = ['piano', 'harpsichord', 'clavichord']

HEADER_KEY_STRINGS = ['mutopiatitle',
                      'mutopiaopus',
                      'mutopiacomposer',
                      'mutopiainstrument']


##############################################################################


def load_ly_lines(filename):
    with codecs.open(filename, 'r', 'utf-8') as hdl:
        lines = [l for l in hdl]
    return lines


def is_include(line):
    tokens = line.strip().split()
    if INCLUDE_TOKEN in tokens:
        return True
    else:
        return False


def parse_include_link(line, current_abs_path):
    """Returns the absolute path to the file included on the line.

    :param line: A line with an ``\include`` statement. Assumptions

    :param current_abs_path: The absolute path to the file which called
        the ``\include``.

    :return: Abs. path to included file.
    """
    tokens = line.strip().split()
    if len(tokens) < 2:
        raise ValueError('Should process \\include statement, but the line'
                         ' does not have enough tokens:\n\t{0}'.format(tokens))
    f_token = None
    for i, t in enumerate(tokens[:-1]):
        if t == INCLUDE_TOKEN:
            f_token = tokens[i+1]
            break

    if f_token is None:
        raise ValueError('Should process \\include statment, but the line'
                         ' does not contain any:\n\t{0}'.format(tokens))

    fname_rel = f_token[1:-1]

    current_abs_dir = os.path.dirname(current_abs_path)
    include_path = os.path.normpath(os.path.join(current_abs_dir, fname_rel))
    abs_include_path = os.path.abspath(include_path)
    return abs_include_path


def is_piano_solo(ly_data):
    """Finds out whether the piece is a piano solo piece.
    If something non-standard happens, is careful & assumes
    it is *not* a piano solo piece."""
    instrument_lines = [l.strip() for l in ly_data
                        if l.strip().startswith('mutopiainstrument')]
    if len(instrument_lines) > 1:
        logging.warning('More than one mutopiainstrument line!'
                        '\n{0}'.format(instrument_lines))
        return False
    if len(instrument_lines) == 0:
        logging.warn('No instrument name found, cannot guess'
                     ' whether it is a piano solo! Returning False.')
        return False

    instrument_line = instrument_lines[0]
    try:
        _, _, instrument_str = instrument_line.split(None, 2)
    except ValueError:
        logging.warn('Cannot properly parse instrument name:\n\t{0}'
                     ''.format(instrument_line))
        return False

    # Normalize instrument names
    instrument_str = instrument_str[1:-1].strip()
    instrument_str = re.sub('[^a-zA-Z]', ' ', instrument_str)
    instrument_tokens = instrument_str.lower().split()

    non_piano_instruments = [t for t in instrument_tokens
                             if t not in PIANO_LIKE_INSTRUMENTS]
    if len(non_piano_instruments) > 0:
        return False
    else:
        return True


def load_header(ly_data):
    """Loads the header dict from the given *.ly lines.
    NOT resistant to malformed header lines!"""
    header = {}

    def _get_header_value(line):
        tokens = line.strip().split('"')
        value = tokens[-2]
        return value

    for l in ly_data:
        for hkey in HEADER_KEY_STRINGS:
            if l.strip().startswith(hkey):
                hval = _get_header_value(l)
                header[hkey] = hval

    return header

##############################################################################


class MutopiaOpus(object):
    """Represents a Mutopia opus. This is a collection of one
    or more pieces.

    Each piece has a top-level *.ly file.

    The purpose of this class is to provide a useful abstraction
    above the non-standardized Mutopia filesystem for individual
    pieces. In order to discover what the top-level files are,
    we unfortunately need to resolve all the includes within an opus.
    Therefore, we need this abstraction to keep track of how the
    lilypond files have been included into one another.
    """
    def __init__(self, root):
        """

        :param root: Root directory of the Mutopia opus. This is
            always a direct child directory of a composer directory,
            which in turn is directly in the ``ftp/`` directory
            of the Mutopia repo.
        """
        if not os.path.isdir(root):
            raise ValueError('Cannot initialize MutopiaOpus with'
                             ' non-existent directory: {0}'
                             ''.format(root))

        self.root = os.path.normpath(os.path.abspath(root))

        self.ly_files = self.discover_lys()

        self.includes_dict = collections.defaultdict(list)
        self.reverse_includes_dict = collections.defaultdict(list)
        includes = self.discover_includes()
        for p, ch in includes:
            self.includes_dict[p].append(ch)
            self.reverse_includes_dict[ch].append(p)

        # One header file per piece.
        self.header_files = self.discover_headers()

        self.piece_files, self.piece_headers = self.find_piece_files()
        self.piece_names = self.generate_piece_names()

    @property
    def n_pieces(self):
        return len(self.piece_files)

    def discover_lys(self):
        """Returns a list of *.ly files inside the opus, as absolute paths.
        Finds also *.ily and *.lyi files."""
        cmd = 'find {0} -type f' \
              ' \( -name *.ly -o -name *.ily -o -name *.lyi \)' \
              ''.format(self.root)
        rel_paths = subprocess.check_output(shlex.split(cmd)).split()
        abs_paths = [os.path.join(self.root, ly_file) for ly_file in rel_paths]
        return abs_paths

    def discover_includes(self):
        """Returns a list of tuples ``(parent, child)``. The ``parent``
        and ``child``entries correspond to the entries in ``self.ly_files``.

        :param reverse: If set, will instead return ``(child, parent)`` tuples.
        """
        output = []
        for ly_file in self.ly_files:
            ly_data = load_ly_lines(ly_file)
            for l in ly_data:
                if is_include(l):
                    abs_incl_file = parse_include_link(l,
                                                       current_abs_path=ly_file)
                    if not os.path.isfile(abs_incl_file):
                        logging.info('Opus {0}: Found a non-existent import: {1}'
                                     ''.format(self.root, abs_incl_file))
                    elif abs_incl_file not in self.ly_files:
                        logging.warn('Opus {0}: Found an import which is not'
                                     ' within the opus: {1}'
                                     ''.format(self.root, abs_incl_file))
                    else:
                        output.append((ly_file, abs_incl_file))

        return output

    def discover_headers(self):
        """Returns a list of file abspaths for Ly files in the opus
        that contain a Mutopia header."""
        header_files = []
        for ly_file in self.ly_files:
            ly_data = load_ly_lines(ly_file)
            for l in ly_data:
                if l.strip() == '':
                    continue

                potential_key = l.strip().replace('=', ' ').split(None, 1)[0]
                if potential_key in HEADER_KEY_STRINGS:
                    header_files.append(ly_file)
                    break

        return header_files

    def find_piece_files(self):
        """Returns a list of the top-level piece files. A top-level
        file is a file which, when all of its includes are processed,
        encodes the entire piece.

        The rules for discovering top-level files are:

        * It is a header file, or it includes a header file;
        * It is not included by any other file.

        We also enforce the constraint that each piece has to
        have its own header -- headers cannot be shared between
        pieces. That means: if a header file is included by more
        than one file, we consider it "dead": it will not contribute
        to any piece.

        :returns: A set of top-level files (abs. paths).
        """
        piece_files = []
        piece_header_files = []
        for ly_file in self.header_files:

            # The easy situation: header file, never included in anything
            if ly_file not in self.reverse_includes_dict:
                piece_files.append(ly_file)
                piece_header_files.append(ly_file)  # ...Its own header
                continue

            # The difficult situation: header file is included in something,
            # which may again be included in something else, etc.
            # THERE IS NO GUARD AGAINST CYCLES.
            stack = [ly_file]
            while len(stack) > 0:
                current_ly_file = stack.pop()

                # Current file is top-level if it is not included anywhere:
                if current_ly_file not in self.reverse_includes_dict:
                    piece_files.append(current_ly_file)
                    piece_header_files.append(ly_file)
                    break

                parents = self.reverse_includes_dict[current_ly_file]
                if len(parents) > 1:
                    logging.warn('Header file {0} included in multiple'
                                 ' files, invalid as piece.'.format(ly_file))
                    break

                parent = parents[0]
                # Child has a header
                if (parent in self.header_files) \
                        and (current_ly_file in self.header_files):
                    piece_files.append(current_ly_file)
                    piece_header_files.append(ly_file)
                    break

                stack.extend(parents)

        # Check for multi-level piece files:
        # piece files that include other piece files
        filtered_piece_files, filtered_piece_header_files = [], []
        for p, h in zip(piece_files, piece_header_files):
            incls = self.includes_dict[p]
            piece_incls = [i for i in incls if i in piece_files]
            if len(piece_incls) == 0:
                filtered_piece_files.append(p)
                filtered_piece_header_files.append(h)

        return filtered_piece_files, filtered_piece_header_files

    def generate_piece_names(self):
        """Generates for each piece in the opus its target name.
        The target name has the format:

           mutopiacomposer_mutopiaopus_basename

        This is derived from the piece's header, and the basename
        of its file.
        """
        names = []
        for pfile, pheader in zip(self.piece_files, self.piece_headers):
            # h_data = load_ly_lines(pheader)
            # header = load_header(h_data)
            basename = os.path.splitext(
                            os.path.basename(
                                os.path.normpath(pfile)))[0]

            # Exploiting the Mutopia composer/opus/ file structure here:
            composer = os.path.basename(os.path.dirname(
                os.path.normpath(self.root)))
            opus = os.path.basename(os.path.normpath(self.root))
            name = '{0}__{1}__{2}'.format(composer,
                                          opus,
                                          basename)
            names.append(name)
        return names


class MutopiaComposer(object):
    """Represents the works of a composer. Exposes a list of all the piano
    pieces by the composer."""
    def __init__(self, root):
        if not os.path.isdir(root):
            raise ValueError('Cannot initialize MutopiaComposer with'
                             ' non-existent directory: {0}'
                             ''.format(root))
        self.root = root
        self.opera = self.collect_opera()
        self.pieces, self.piece_names = self.collect_pieces()
        self.piano_pieces = [p for p in self.pieces
                             if is_piano_solo(load_ly_lines(p))]
        self.piano_piece_names = [n
                                  for p, n in zip(self.pieces,
                                                  self.piece_names)
                                  if p in self.piano_pieces]

    def collect_opera(self, permissive=True):
        """Collects all the Opus objects (correct plural: Opera) by
        the composer."""
        opera = []
        for opus_dir in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root, opus_dir)):

                ### DEBUG
                print('\n\nBuilding opus: {0}'.format(os.path.join(self.root, opus_dir)))
                try:
                    opus = MutopiaOpus(root=os.path.join(self.root, opus_dir))
                    opera.append(opus)
                except Exception as e:
                    if permissive:
                        print('\t\t...could not build opus; skipping.')
                    else:
                        raise e
        return opera

    def collect_pieces(self):
        """Collects the top-level files for all Opera by the composer."""
        pieces = []
        piece_names = []
        for o in self.opera:
            pieces.extend(o.piece_files)
            piece_names.extend(o.piece_names)
        return pieces, piece_names


class MutopiaCorpus(object):
    def __init__(self, root):
        self.root = root

        self.composers = self.collect_composers()

        self.piano_pieces, self.piano_piece_names = self.collect_piano_pieces()

    def collect_composers(self):
        composers = []
        for c in os.listdir(self.root):
            if c.startswith('.'):
                continue
            if os.path.isdir(os.path.join(self.root, c)):
                composer = MutopiaComposer(os.path.join(self.root, c))
                composers.append(composer)
        return composers

    def collect_piano_pieces(self):
        piano_pieces = []
        piano_piece_names = []
        for c in self.composers:
            piano_pieces.extend(c.piano_pieces)
            piano_piece_names.extend(c.piano_piece_names)
        return piano_pieces, piano_piece_names

