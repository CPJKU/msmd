#!/usr/bin/env python
"""This is a script that preprocesses Mutopia LilyPond source files.
The following tasks are carried out:

* Processing ``include`` functions
* Stripping away ``unfoldRepeats``
* Ensure spacing through ``paper`` (?)

This is necessary upon *extraction of the Mutopia file* to the Sheet
Manager dataset, before the Sheet Manager processing pipeline does anything.
"""

from __future__ import print_function, unicode_literals

import argparse
import codecs
import logging
import os
import pickle
import re
import time

from sheet_manager.mutopia.parsing import PIANO_LIKE_INSTRUMENTS, is_include, parse_include_link, load_ly_lines, \
    MutopiaCorpus

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################


def process_includes(lines, filename, join=True):
    """Creates the LilyPond string corresponding to the file with \include
     statements processed. Can deal with recursive includes, but cannot detect
     cyclical includes."""
    abs_path = os.path.normpath(os.path.abspath(filename))

    output_lines = []

    # Load this file
    if lines is None:
        with codecs.open(abs_path, 'r', 'utf-8') as hdl:
            lines = [l for l in hdl]

    # If \include statement on the line:
    for l in lines:
        if is_include(l):
            included_filename = parse_include_link(l,
                                                   current_abs_path=abs_path)

            if not os.path.isfile(included_filename):
                logging.warning('Unresolvable include: {0}'.format(l))
                output_lines.append(l)
                continue

            included_lines = process_includes(lines=None,
                                              filename=included_filename,
                                              join=False)
            output_lines.extend(included_lines)
        else:
            output_lines.append(l)

    if join:
        # Assumes we have newlines.
        output = ' '.join(output_lines)
    else:
        output = output_lines

    return output


def no_unfold_repeats(lines):
    """Removes all ``unfoldRepeats`` tokens."""
    output = []
    for i, l in enumerate(lines):
        output_line = re.sub('\\\\unfoldRepeats', '', l)
        if l != output_line:
            logging.info('Found unfoldRepeats on line {0}'.format(i))
        output.append(output_line)
    return output


def process_file(ly_file):
    ly_data = load_ly_lines(ly_file)
    with_includes = process_includes(ly_data,
                                     filename=ly_file,
                                     join=False)
    no_unfolds = no_unfold_repeats(with_includes)
    return no_unfolds

##############################################################################


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--input_file', action='store', default=None,
                        help='The LilyPond file that should be preprocessed.'
                             ' If this is set, will only process this file.'
                             ' For batch-processing the Mutopia dataset, use'
                             ' --mutopia_root.')
    parser.add_argument('-o', '--output_file', action='store',
                        help='The output where the preprocessed file should be'
                             ' stored. Only makes sense with --input_file.')

    parser.add_argument('--mutopia_root', action='store',
                        help='The root of the Mutopia repo ftp/ area.')
    parser.add_argument('--output_root', action='store',
                        help='The target PiPeS dataset directory. Piece'
                             ' directories will be created here.')

    parser.add_argument('--export_corpus', action='store',
                        help='Pickle the loaded corpus object to this'
                             ' file. Then, you can use --load_corpus'
                             ' so that it does not have to be rebuilt.')
    parser.add_argument('--load_corpus', action='store',
                        help='Load the corpus object from this file,'
                             ' built previously with --export_corpus.')

    parser.add_argument('--force', action='store_true',
                        help='If set, will overwrite existing pieces with the'
                             ' same names in the output directory.')
    parser.add_argument('--first_k', action='store', type=int, default=0,
                        help='If set, only process the first K pieces.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    if args.input_file is not None:
        output_lines = process_file(args.input_file)

        with codecs.open(args.output_file, 'w', 'utf-8') as hdl:
            for l in output_lines:
                hdl.write(l)
            hdl.write('\n')

    else:
        if args.load_corpus:
            with open(args.load_corpus, 'rb') as hdl:
                corpus = pickle.load(hdl)
        else:
            corpus = MutopiaCorpus(args.mutopia_root)

        if args.export_corpus:
            with open(args.export_corpus, 'wb') as hdl:
                pickle.dump(corpus, hdl, protocol=pickle.HIGHEST_PROTOCOL)

        n_processed = 0
        for i, (piece, name) in enumerate(zip(corpus.piano_pieces,
                                              corpus.piano_piece_names)):

            output_dir = os.path.join(args.output_root, name)
            output_filename = os.path.join(output_dir, name + '.ly')
            if os.path.isfile(output_filename):
                if not args.force:
                    logging.info('Piece already exported, skipping: {0}'
                                 ''.format(name))
                    continue

            _piece_start_time = time.clock()

            processed_lines = process_file(piece)

            if not os.path.isdir(output_dir):
                os.mkdir(output_dir, 0o755)
            with codecs.open(output_filename, 'w', 'utf-8') as hdl:
                hdl.write('\n'.join(processed_lines))
                hdl.write('\n')

            n_processed += 1

            _now = time.clock()
            print('({0}) piece: {1}\tTime: {2:.2f} s\tTotal: {3:.2f} s'
                  ''.format(n_processed, name,
                            _now - _piece_start_time,
                            _now - _start_time))

            if args.first_k:
                if n_processed >= args.first_k:
                    break

    _end_time = time.clock()
    logging.info('process_mutopia.py done in {0:.3f} s'
                 ''.format(_end_time - _start_time))


##############################################################################


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
