"""This module implements the abstraction over a given piece
of music."""
from __future__ import print_function

import logging
import os
import shutil

import sys

import time
import yaml

from msmd.data_model.performance import Performance
from msmd.data_model.score import Score
from msmd.data_model.util import SheetManagerDBError, MSMDMetadataMixin

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


class Piece(MSMDMetadataMixin):
    """This class represents a single piece. The piece with all its
    scores, performances, encodings, etc. lives in the filesystem
    in a directory; this class is just an abstraction to make manipulating
    the piece more comfortable.

    Attributes
    ----------

    * ``name``: a unique identifier of the piece within the collection.
    * ``folder``: the path to the folder of the piece.
    * ``collection_root``: the Piece remembers the collection it is a part of.
    * ``metadata``: a dict of arbitrary descriptors. Expects a ``meta.yml``
      file in the ``piece.folder`` directory.
    * ``encodings``: a dict of the available encoding files. The keys
      are encoding shortcuts (mxml, ly, midi, mei, ...). One of these
      files should always be available. When the piece is initialized,
      one of the available encodings must be selected as the authority
      encoding, from which everything else (scores, performances, ...)
      is derived.
    * ``performances``: a dict of available performances (directories).
      Keys are performance names, values are paths.
    * ``scores``: a dict of available scores (directories). Keys are
      score names, values are paths.

    All paths stored in the attributes include the ``collection_root``
    path. If the root is ``~/baroque`` and there is a piece in the
    collection called ``reichenauer_sonata-1_allegro``, then the path
    stored in ``folder`` will be ``~/baroque/reichenauer_sonata-1_allegro``.
    This implies: if ``collection_root`` is a relative path, all
    the other paths will be relative to the same point in the filesystem
    to which the ``collection_root`` is relative.

    Getting performances and scores
    -------------------------------

    The Piece only stores the paths. In order to load the actual
    ``Performance`` and ``Score`` objects, you need to use the
    ``load_performance()``, ``load_all_performances()`` methods (and
    their ``load_score()`` analogies).

    """
    DEFAULT_META_FNAME = 'meta.yml'
    AVAILABLE_AUTHORITIES = ['mxml', 'ly', 'midi', 'mei']

    def __init__(self, name, root, authority_format='ly'):
        """Initialize the Piece.

        :param root: The root directory of the collection. Contains
            the directories for individual pieces.

        :param piece_name: The directory that contains everything associated
            with the given piece: encodings, scores, performances, etc.

        :param authority_encoding: Each piece is at its core represented
            by the *authority encoding* (term borrowed from librarians).
            This is the "unmovable truth" for the piece; if performances
            or scores are generated automatically for the piece (which is
            usually the case), the sheet manager workflow will be based
            on this value. See ``AVAILABLE_ENCODINGS`` class attribute
            for a list of encodings which it is possible to specify.
        """
        super(Piece, self).__init__()

        if not os.path.isdir(root):
            raise SheetManagerDBError('Collection root directory does not'
                                      ' exist: {0}'.format(root))

        piece_folder = os.path.join(root, name)
        if not os.path.isdir(piece_folder):
            raise SheetManagerDBError('Piece {0} in collection {1} does'
                                      ' not exist'.format(name, root))

        if authority_format not in self.AVAILABLE_AUTHORITIES:
            raise ValueError('Authority format not supported: {0}'
                             ''.format(authority_format))

        self.name = name
        self.folder = os.path.normpath(piece_folder)
        self.collection_root = os.path.normpath(root)

        self.performance_dir = os.path.join(self.folder, 'performances')
        self.score_dir = os.path.join(self.folder, 'scores')
        self._ensure_piece_structure()

        self.metadata = self.load_metadata()

        self.encodings = self.collect_encodings()
        if authority_format not in self.encodings:
            raise SheetManagerDBError('Piece {0} in collection {1} does'
                                      ' not have the requested authority'
                                      ' encoding {2}. (Available encodings:'
                                      ' {3}'.format(self.name, root,
                                                    authority_format,
                                                    self.encodings.values()))
        self.authority_format = authority_format
        self.authority = self.encodings[authority_format]

        self.performances = self.collect_performances()
        self.scores = self.collect_scores()

    @property
    def available_performances(self):
        return sorted(self.performances.keys())

    @property
    def available_scores(self):
        return sorted(self.scores.keys())

    @property
    def default_score_name(self):
        return self.name + '_' + self.authority_format

    @property
    def metadata_folder(self):
        # Alias for the MSMDMetadataMixin
        return self.folder

    @property
    def composer(self):
        return self.composer_name_from_piece_name(self.name)

    @staticmethod
    def composer_name_from_piece_name(piece_name):
        """Based on the piece name, extracts the composer name."""
        separator = '__'
        composer = piece_name.split(separator)[0]
        return composer

    @property
    def seconds(self):
        """How many seconds does this piece take by default?"""
        # Select performance with natural tempo
        nat_perf = None
        for perf in self.available_performances:
            if '1000' in perf:
                nat_perf = perf
                break

        if nat_perf is None:
            raise ValueError('Cannot yet compute # seconds from non-natural'
                             ' performance.')

        # Return its length
        p = self.load_performance(nat_perf,
                                  require_midi=False, require_audio=False)
        return p.length_in_seconds()

    @property
    def n_pages(self):
        score = self.load_score(self.scores.keys()[0])
        return score.n_pages

    def _ensure_piece_structure(self):
        """Creates the basic expected directory structure."""
        if not os.path.isdir(self.performance_dir):
            os.mkdir(self.performance_dir)
        if not os.path.isdir(self.score_dir):
            os.mkdir(self.score_dir)

    def load_score(self, score_name):
        self.update()
        if score_name not in self.scores:
            raise SheetManagerDBError('Piece {0} in collection {1} does'
                                      ' not have a score with name {2}.'
                                      ' Available scores: {3}'
                                      ''.format(self.name, self.collection_root,
                                                score_name,
                                                self.available_scores))
        score_dir = self.scores[score_name]
        score = Score(folder=score_dir, piece_name=self.name)
        return score

    def load_all_scores(self, score_name):
        """Returns a list of all the available Scores."""
        return [self.load_score(s) for s in self.available_scores]

    def load_performance(self, performance_name, **perf_kwargs):
        """Creates a ``Performance`` object for the given performance
        and returns it. You can pass Performance initialization kwargs."""
        self.update()
        if performance_name not in self.performances:
            raise SheetManagerDBError('Piece {0} in collection {1} does'
                                      ' not have a performance with name {2}.'
                                      ' Available performances: {3}'
                                      ''.format(self.name, self.collection_root,
                                                performance_name,
                                                self.available_performances))
        performance_dir = self.performances[performance_name]
        performance = Performance(folder=performance_dir,
                                  piece_name=self.name,
                                  **perf_kwargs)
        return performance

    def load_all_performances(self, **perf_kwargs):
        """Returns a list of all the available Performances. You can pass
        Performance initialization kwargs."""
        return [self.load_performance(p, **perf_kwargs)
                for p in self.available_performances]

    def update(self):
        """Refreshes the index of available performances
        and scores."""
        self.encodings = self.collect_encodings()
        self._set_authority(self.authority_format)

        self._ensure_piece_structure()
        self.performances = self.collect_performances()
        self.scores = self.collect_scores()

    def _set_authority(self, authority_format):
        """Sets the authority to the selected format. Don't do this
        unless you are sure what you are doing. If you really need
        to derive something in the piece from different authority
        encodings, consider initializing another ``Piece`` instance."""
        if authority_format not in self.AVAILABLE_AUTHORITIES:
            raise ValueError('Authority format not supported: {0}'
                             ''.format(authority_format))
        if authority_format not in self.encodings:
            raise SheetManagerDBError('Piece {0} in collection {1} does'
                                      ' not have the requested authority'
                                      ' encoding {2}. (Available encodings:'
                                      ' {3}'.format(self.name,
                                                    self.collection_root,
                                                    authority_format,
                                                    self.encodings.values()))
        self.authority_format = authority_format
        self.authority = self.encodings[authority_format]

    def collect_performances(self):
        """Collects a dict of the available performances. Keys
        are performance names (corresponding to directory naes
        in the ``self.performance_dir`` directory), values are
        the paths to these directories."""
        performances = {p: os.path.join(self.performance_dir, p)
                        for p in os.listdir(self.performance_dir)
                        if os.path.isdir(os.path.join(self.performance_dir, p))}
        return performances

    def collect_scores(self):
        """Collects a dict of the available scores. Keys
        are score names (corresponding to directory naes
        in the ``self.score_dir`` directory), values are
        the paths to these directories."""
        scores = {s: os.path.join(self.score_dir, s)
                  for s in os.listdir(self.score_dir)
                  if os.path.isdir(os.path.join(self.score_dir, s))}
        return scores

    def collect_encodings(self):
        """Collects various encodings that SheetManager can deal with:

        * MusicXML (*.xml)
        * LilyPond (*.ly)
        * Normalize LilyPond (*.norm.ly)
        * MIDI (*.midi)
        * MEI (*.mei)

        Out of these, the authority encoding can be chosen, but it has
        to be one of a more restricted set, as specified by the
        ``AVAILABLE_AUTHORITIES`` class attribute.

        :returns: A dict of the encoding files. The keys are the encoding
            names: ``mxml``, ``ly``, ``norm.ly``, ``midi``, ``mei``
            (if the corresponding files are available).
        """
        encodings = dict()

        mxml = os.path.join(self.folder, self.name + '.xml')
        if os.path.isfile(mxml):
            encodings['mxml'] = mxml

        ly = os.path.join(self.folder, self.name + '.ly')
        if os.path.isfile(ly):
            encodings['ly'] = ly

        normalized_ly = os.path.join(self.folder, self.name + '.norm.ly')
        if os.path.isfile(normalized_ly):
            encodings['norm.ly'] = normalized_ly

        midi = os.path.join(self.folder, self.name + '.mid')
        if not os.path.isfile(midi):
            midi += 'i'
        if os.path.isfile(midi):
            encodings['midi'] = midi

        mei = os.path.join(self.folder, self.name + '.mei')
        if os.path.isfile(mei):
            encodings['mei'] = mei

        return encodings

    # def load_metadata(self):
    #     """Loads arbitrary YAML descriptors with the default name (meta.yml)."""
    #     metafile = os.path.join(self.folder, self.DEFAULT_META_FNAME)
    #     if not os.path.isfile(metafile):
    #         logging.info('Piece {0} has no metadata file: {1}'
    #                      ''.format(self.name, self.DEFAULT_META_FNAME))
    #         return dict()
    #
    #     with open(metafile, 'r') as hdl:
    #         metadata = yaml.load_all(hdl)
    #
    #     return metadata
    #
    # def dump_metadata(self):
    #     """Dumps the current metadata of the piece. Overwrites the current
    #     metafile.
    #
    #     Because the performances are not loaded, we cannot dump their
    #     metadata from here; it needs to be done on an as-needed basis.
    #     """
    #     if not self.metadata:
    #         return
    #
    #     metafile = os.path.join(self.folder, self.DEFAULT_META_FNAME)
    #     with open(metafile, 'w') as hdl:
    #         yaml.dump(self.metadata, hdl)

    def clear(self):
        """Removes all scores, performacnes, and non-authority
        encodings. Use this very carefully!"""
        self.clear_performances()
        self.clear_scores()
        for e in self.encodings:
            if e != self.authority_format:
                os.unlink(self.encodings[e])

    def clear_performances(self):
        """Remove all performances. Use this carefully!"""
        self.update()
        for p in self.performances:
            self.remove_performance(p)

    def remove_performance(self, name):
        """Removes the given performance folder."""
        self.update()
        if name not in self.performances:
            logging.warn('Piece {0}: trying to remove performance {1},'
                         ' but it does not exist!'.format(self.name, name))
            return

        shutil.rmtree(self.performances[name])
        self.update()

    def add_performance(self, name, audio_file=None, midi_file=None,
                        overwrite=False):
        """Creates a new performance in the piece from existing audio
        and optionally MIDI files.

        :param name: Name of the new performance. The performance folder
            will have this name, and the performance audio (and midi) file
            names will be derived from this name by simply copying the format
            suffix from the ``audio_file`` and ``midi_file`` arguments.

        :param audio_file: The audio file for the performance. Will be copied
            into the newly created performance directory, with the filename
            derived as the `name`` plus the format suffix.

        :param midi_file: The performance MIDI. Optional. Will be copied
            into the newly created performance directory. Same name convention
            as for ``audio_file``.

        :param overwrite: If true, if a performance with the given ``name``
            exists, will delete it.
        """
        if (audio_file is None) and (midi_file is None):
            raise ValueError('At least one of audio and midi files'
                             ' has to be supplied to create a performance.')
        if name in self.performances:
            if overwrite:
                logging.info('Piece {0}: performance {1} already exists,'
                             ' overwriting!'.format(self.name, name))
                time.sleep(5)
                self.remove_performance(name)
            else:
                raise SheetManagerDBError('Piece {0}: performance {1} already'
                                          ' exists!'.format(self.name, name))
        new_performance_dir = os.path.join(self.performance_dir, name)

        # This part should be refactored as performance.build_performance()
        os.mkdir(new_performance_dir)

        audio_fmt = os.path.splitext(audio_file)[-1]
        performance_audio_filename = os.path.join(new_performance_dir,
                                           name + audio_fmt)
        shutil.copyfile(audio_file, performance_audio_filename)

        if midi_file:
            midi_fmt = os.path.splitext(midi_file)[-1]
            performance_midi_filename = os.path.join(new_performance_dir,
                                                     name + midi_fmt)
            shutil.copyfile(midi_file, performance_midi_filename)

        self.update()

        # Test-load the Performance. Ensures folder structure initialization.
        _ = self.load_performance(name,
                                  require_audio=(audio_file is not None),
                                  require_midi=(midi_file is not None))

    def clear_scores(self):
        """Removes all scores of the piece. Use this carefully!"""
        self.update()
        for s in self.scores:
            self.remove_score(s)

    def remove_score(self, name):
        """Removes the given score folder."""
        self.update()
        if name not in self.scores:
            logging.warn('Piece {0}: trying to remove score {1},'
                         ' but it does not exist!'.format(self.name, name))
            return

        shutil.rmtree(self.scores[name])
        self.update()

    def add_score(self, name, pdf_file, overwrite=False):
        """Creates a new score in the piece from an existing PDF file.

        :param name: Name of the new score. The score folder
            will have this name, and the score PDF file
            names will be derived from this name by simply
            adding the ``.pdf`` suffix to the name.

        :param pdf_file: The PDF authority for the given score. Will be copied
            into the newly created score directory, with the filename
            derived as the `name`` plus the ``.pdf`` suffix. Required.

        :param overwrite: If true, if a score with the given ``name``
            exists, will delete it and overwrite with the new one.
        """
        if name in self.scores:
            if overwrite:
                logging.info('Piece {0}: performance {1} already exists,'
                             ' overwriting!'.format(self.name, name))
                time.sleep(5)
                self.remove_score(name)
            else:
                raise SheetManagerDBError('Piece {0}: performance {1} already'
                                          ' exists!'.format(self.name, name))
        new_score_dir = os.path.join(self.score_dir, name)

        os.mkdir(new_score_dir)

        score_pdf_filename = os.path.join(new_score_dir, name + '.pdf')
        shutil.copyfile(pdf_file, score_pdf_filename)

        self.update()

        # Test-load the Score. Ensures folder structure initialization.
        _ = self.load_score(name)
