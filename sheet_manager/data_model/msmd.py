"""This module implements a class that..."""
from __future__ import print_function

import collections
import os
import pprint

import yaml

from sheet_manager.data_model.piece import Piece
from sheet_manager.data_model.util import SheetManagerDBError
from sheet_manager.utils import aggregate_dicts, reduce_dicts

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


class MSMD(object):
    """The MSMD class is an abstraction over the MSMD dataset."""

    def __init__(self, folder):
        """Initialize the dataset object."""
        if not os.path.isdir(folder):
            raise SheetManagerDBError('MSMD root does not exist: {0}'
                                      ''.format(folder))
        self.folder = folder

        self.pieces = self.collect_pieces()

    def update(self):
        self.pieces = self.collect_pieces()

    def collect_pieces(self, composers=None):
        """Collects a dict of the available pieces. Keys
        are piece names (corresponding to directory naes
        in the ``self.folder`` directory), values are the paths
        of the pieces.

        :param composers: A list of composer names according to Mutopia
            (so, e.g., ``['BachCPE', 'Schumann']``). If given, will
            only collect pieces by these composers.
        """
        pieces = collections.OrderedDict()
        if composers is not None:
            composers = set(composers)
        for p in sorted(os.listdir(self.folder)):
            piece_dir = os.path.join(self.folder, p)

            if not os.path.isdir(piece_dir):
                continue

            if composers is not None:
                if Piece.composer_name_from_piece_name(p) not in composers:
                    continue

            pieces[p] = piece_dir

        return pieces

    def load_piece(self, name):
        """Returns a Piece object."""
        if name not in self.pieces:
            raise SheetManagerDBError('Requested piece does not exist: {0}'
                                      ''.format(name))

        piece = Piece(name, root=self.folder)
        return piece

    def stats(self, pieces=None):
        """Returns an aggregate statistic over the given pieces.
        The aggregation will be over pieces: the ``aln_piece_stats``
        metadata entry is used, so that the alignments don't have
        to be recomputed.

        :param pieces: A list of piece names on which to compute
            aggregate statistics. (See the ``alignment_stats()``
            function for what statistics are computed.)
        """
        if pieces is None:
            pieces = self.pieces.keys()

        _metadata_stats_key = 'aln_piece_stats'

        all_piece_stats = []
        for piece_name in pieces:
            piece = self.load_piece(piece_name)
            if piece.metadata['processed']:
                if piece.metadata['aligned_well']:

                    if _metadata_stats_key not in piece.metadata:
                        # print('Metadata key missing! Piece: {0}, Metadata: {1}'
                        #       ''.format(piece.name, pprint.pformat(piece.metadata)))
                        continue

                    stats = piece.metadata[_metadata_stats_key]

                    # Compute seconds
                    stats['seconds'] = piece.seconds

                    all_piece_stats.append(stats)


        aggregate = reduce_dicts(all_piece_stats, fn=sum)
        # average = average_dicts(all_piece_stats)
        return aggregate

    def stats_on_split(self, split_file):
        """Computes the aggregate statistics from the given split file."""
        with open(split_file, 'rb') as hdl:
            splits_dict = yaml.load(hdl)

        train_pieces = splits_dict['train']
        train_stats = self.stats(train_pieces)

        valid_pieces = splits_dict['valid']
        valid_stats = self.stats(valid_pieces)

        test_pieces = splits_dict['test']
        test_stats = self.stats(test_pieces)

        return train_stats, valid_stats, test_stats
