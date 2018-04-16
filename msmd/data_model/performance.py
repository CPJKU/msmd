"""This module implements a class that represents a performance
of a given piece."""
from __future__ import print_function

import logging
import os

import numpy
import yaml

from msmd.data_model.util import MSMDDBError, path2name, MSMDMetadataMixin

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


class Performance(MSMDMetadataMixin):
    """The Performance class represents one performance of a piece
    (live or synthetic). Each performance has:

    * audio (this is the authority file for a performance)
    * midi (technically not necessary, but esp. in performances synthesized
      by Sheet Manager, it will always be there. If it is there, it also
      acts as an authority for MIDI-based features, since extracting MIDI
      from an audio is a really hard problem...)
    * features (various numpy arrays in the features/ subdirectory
      of the performance directory).
    """
    DEFAULT_META_FNAME = 'meta.yml'
    # AUDIO_EXTENSIONS = ['flac', 'wav', 'mp3', 'ogg']
    AUDIO_NAMING_SEPARATOR = '_'

    def __init__(self, folder, piece_name, audio_fmt='flac',
                 require_audio=True, require_midi=True):
        """Initialize Performance.

        :param audio_fmt: The audio of the performance is expected
            to have this format.
        """
        super(Performance, self).__init__()

        if not os.path.isdir(folder):
            raise MSMDDBError('Performance initialized with'
                                      ' non-existent directory: {0}'
                                      ''.format(folder))
        self.folder = folder
        name = path2name(folder)
        self.name = name
        self.piece_name = piece_name

        self.metadata = self.load_metadata()

        if audio_fmt.startswith('.'):
            audio_fmt = audio_fmt[1:]
        self.audio_fmt = audio_fmt

        self.audio = self.discover_audio(required=require_audio)
        self.audio_name = None
        if self.audio:
            self.audio_name = path2name(self.audio)

        self.midi = self.discover_midi(required=require_midi)

        self.features_dir = os.path.join(self.folder, 'features')
        self._ensure_features_dir()

        self.features = self.collect_features()

    @property
    def metadata_folder(self):
        return self.folder

    def update(self):
        self.audio = self.discover_audio()
        self.audio_name = path2name(self.audio)
        self.midi = self.discover_midi()
        self.update_features()

    def update_features(self):
        self._ensure_features_dir()
        self.features = self.collect_features()

    def add_feature(self, feature, suffix, overwrite=False):
        """Adds the given feature to the Performance. Enforces
        the suffix naming conventions: if you save computed features
        to the performance using this method, you are guaranteed
        to be able to find them later using the ``_load_feature_by_suffix()``
        method.

        :param suffix: The distinguishing of the feature, e.g. ``midi``
            for the MIDI matrix feature, or ``spec`` for the spectrogram.
            Do not supply the separator; the method takes care of using
            the appropriate separator to incorporate the suffix into the
            filename.

            The file format also needs to be given. Currently, Performances
            expect features to be numpy arrays, so the suffix should end
            with ``.npy``.
        """
        if not suffix.endswith('.npy'):
            logging.warn('Adding a feature with an unexpected suffix: {0}'
                         ''.format(suffix))

        if not isinstance(feature, numpy.ndarray):
            raise TypeError('Features must be numpy arrays! Got feature'
                            ' of type: {0}'.format(type(feature)))

        # Normalizing the suffix
        while suffix.startswith(self.AUDIO_NAMING_SEPARATOR):
            suffix = suffix[1:]
        feature_name = self.AUDIO_NAMING_SEPARATOR.join([self.audio_name,
                                                         suffix])
        if feature_name in self.features:
            if overwrite:
                logging.info('Performance {0}: overwriting feature {1}!'
                             ''.format(self.folder, feature_name))
            else:
                logging.warn('Performance {0}: feature {1} already exists!'
                             ' Not added.'.format(self.folder, feature_name))
                return

        feature_path = os.path.join(self.features_dir, feature_name)
        numpy.save(feature_path, feature)

        self.update_features()

    def collect_features(self):
        """Returns a dict of the performance features. Any file in the
        ``self.features_dir`` directory is considered to be one features
        file. The keys of the dict are filenames, the values are paths.
        """
        features = {f: os.path.join(self.features_dir, f)
                    for f in os.listdir(self.features_dir)
                    if not f.startswith('.')}
        return features

    def _ensure_features_dir(self):
        if not os.path.isdir(self.features_dir):
            os.mkdir(self.features_dir)

    def _discover_candidate_files(self, suffixes, return_all_candidates=False):
        """Returns a list of the candidate names for MIDI and Audio
        file discovery (and potentially others).

        The discovery looks for a combination of the piece name and
        the performance name in both orders, or in isolation. The
        separator is expected to be an underscore (or you can set
        it to something else in the class attribute
        ``Performance.AUDIO_NAMING_SEPARATOR``).

        :param suffixes: You have to supply
            the file format(s) -- use ``[mid, midi]`` for MIDI file
            discovery, as the default fmts might change. (You can also
            just supply a string if there is only one format you are
            interested in.)

        :param return_all_candidates: If set, will return two lists:
            the first is just the discovered candidates, the second
            list is all the candidate names that were tried.

        :returns: A list of candidate files that exist. If no candidate
            file exists, returns empty list.
        """
        if suffixes is None:
            raise ValueError('Suffixes for file discovery must be specified.')

        if isinstance(suffixes, str):
            suffixes = [suffixes]

        discovered_candidates = []
        all_candidates = []

        for suffix in suffixes:
            if not suffix.startswith('.'):
                suffix = '.' + suffix
            SEP = self.AUDIO_NAMING_SEPARATOR
            suffix_candidate_names = [
                SEP.join([self.piece_name, self.name]) + suffix,
                SEP.join([self.name, self.piece_name]) + suffix,
                self.piece_name + suffix,
                self.name + suffix,
            ]
            candidate_fnames = [os.path.join(self.folder, a)
                                      for a in suffix_candidate_names]
            all_candidates.extend(candidate_fnames)

            for fname in candidate_fnames:
                if os.path.isfile(fname):
                    discovered_candidates.append(fname)

        if return_all_candidates:
            return discovered_candidates, all_candidates
        return discovered_candidates

    def discover_audio(self, required=False):
        """Looks for audio files in the performance directory.

        :param required: If no audio with the format specified for
            the Performance (by default: ``*.flac``) is discovered,
            will raise a ``MSMDDBError``.
        """
        candidate_files = self._discover_candidate_files(self.audio_fmt)
        #
        # SEP = self.AUDIO_NAMING_SEPARATOR
        # suffix = '.' + self.audio_fmt
        # audio_candidate_names = [
        #     SEP.join([self.piece_name, self.name]) + suffix,
        #     SEP.join([self.name, self.piece_name]) + suffix,
        #     self.piece_name + suffix,
        #     self.name + suffix,
        # ]
        # audio_candidate_fnames = [os.path.join(self.folder, a)
        #                           for a in audio_candidate_names]
        # for fname in audio_candidate_fnames:
        #     if os.path.isfile(fname):
        #         return fname

        if len(candidate_files) == 0:
            if required:
                raise MSMDDBError('No audio with requested format {0}'
                                          ' found in performance {1}!'
                                          ''.format(self.audio_fmt, self.folder))
            else:
                return None

        return candidate_files[0]

    def discover_midi(self, required=True):
        """Based on the discovered audio, finds the performance
        MIDI (if available)."""
        midi_fname = None

        if self.audio is None:
            candidate_files, all_candidates = self._discover_candidate_files(['mid', 'midi'],
                                                                             return_all_candidates=True)
            if len(candidate_files) == 0:
                midi_fname = None
            else:
                midi_fname = candidate_files[0]
        else:
            midi_fname = os.path.splitext(self.audio)[0] + '.mid'
            if not os.path.isfile(midi_fname):
                midi_fname += 'i'
            if not os.path.isfile(midi_fname):
                midi_fname = None

        if midi_fname is None:
            if required:
                raise MSMDDBError('No MIDI found in performance {0}!'
                                          ' All candidates: {1}'
                                          ''.format(self.folder, '\n'.join(all_candidates)))

        return midi_fname

    # def load_metadata(self):
    #     """Loads arbitrary YAML descriptors with the default name (meta.yml)."""
    #     metafile = os.path.join(self.folder, self.DEFAULT_META_FNAME)
    #     if not os.path.isfile(metafile):
    #         logging.info('Performance {0} has no metadata file: {1}'
    #                      ''.format(self.name, self.DEFAULT_META_FNAME))
    #         return dict()
    #
    #     with open(metafile, 'r') as hdl:
    #         metadata = yaml.load_all(hdl)
    #
    #     return metadata
    def length_in_seconds(self, FPS=20.0, FIXED_END_LENGTH=2):
        """Computes the length of the performance in seconds.
        Note that it computes this length from the last onset and adds two
        seconds. This is much faster than loading the entire MIDI matrix
        or spectrogram.
        """
        onsets = self.load_onsets()
        last_onset = onsets[-1]
        last_onset_seconds = last_onset / FPS
        total_seconds = last_onset_seconds + FIXED_END_LENGTH
        return total_seconds

    def load_feature(self, feature_name):
        """Loads the feature with the given name, if available
        in self.features. Raises a ValueError otherwise."""
        self.collect_features()
        if feature_name not in self.features:
            raise ValueError('Performance {0}: feature {1} not available!'
                             ' Available feature names: {2}'
                             ''.format(self.name, feature_name,
                                       self.features.keys()))

        if not os.path.isfile(self.features[feature_name]):
            raise MSMDDBError('Performance {0}: feature {1} is'
                                      ' available, but the file {2} does not'
                                      ' exist...?'
                                      ''.format(self.name,
                                                feature_name,
                                                self.features[feature_name]))

        feature = numpy.load(self.features[feature_name])
        return feature

    def load_midi_matrix(self):
        """Shortcut for loading the MIDI matrix feature.
        Expects the feature name ``self.audio_name + '_midi.npy'."""
        return self._load_feature_by_suffix('_midi.npy')

    def load_onsets(self):
        """Shortcut for loading the MIDI matrix feature.
        Expects the feature name ``self.audio_name + '_midi.npy'."""
        return self._load_feature_by_suffix('_onsets.npy')

    def load_spectrogram(self):
        """Shortcut for loading the MIDI matrix feature.
        Expects the feature name ``self.audio_name + '_midi.npy'."""
        return self._load_feature_by_suffix('_spec.npy')

    def load_note_events(self):
        return self._load_feature_by_suffix('notes.npy')

    def _load_feature_by_suffix(self, suffix):
        """Utility function for loading features by suffix naming
        conventions."""
        self.update_features()
        candidate_feature_names = [f for f in self.features
                                   if f.endswith(suffix)]
        if len(candidate_feature_names) == 0:
            raise MSMDDBError('Performance {0}: Feature {1}'
                                      ' not available! Availble feature'
                                      ' names: {2}'.format(self.name, suffix,
                                                           self.features.keys()))
        if len(candidate_feature_names) > 1:
            raise MSMDDBError('Performance {0}: More than one feature'
                                      ' conforms to the suffix {1}: {2}'
                                      ''.format(self.name, suffix,
                                                candidate_feature_names))

        feature_name = candidate_feature_names[0]
        return self.load_feature(feature_name)
