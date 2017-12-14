# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:02:37 2016

@author: matthias

Syntisize audio from midi and extract spectrogram annotations

"""
import logging
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import music21

import madmom.utils.midi as mm_midi
# import madmom.utils.midi_old as mm_midi
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.filters import LogarithmicFilterbank
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor
from madmom.processors import SequentialProcessor

# init signal processing
SAMPLE_RATE = 22050
FRAME_SIZE = 2048
FPS = 20

sig_proc = SignalProcessor(num_channels=1, sample_rate=SAMPLE_RATE)
fsig_proc = FramedSignalProcessor(frame_size=FRAME_SIZE, fps=FPS, origin='future')
spec_proc = FilteredSpectrogramProcessor(LogarithmicFilterbank, num_bands=16, fmin=30, fmax=6000)  # num_bands=24, fmin=30, fmax=8000
log_spec_proc = LogarithmicSpectrogramProcessor()
processor = SequentialProcessor([sig_proc, fsig_proc, spec_proc, log_spec_proc])

colors = ['c', 'm', 'y']


def notes_to_onsets(notes, dt):
    """ Convert sequence of keys to onset frames """

    onsets = []
    for n in notes:
        onset = int(np.ceil(n[0] / dt))
        onsets.append(onset)

    return np.sort(np.asarray(onsets)).astype(np.float32)


def notes_to_matrix(notes, dt):
    """ Convert sequence of keys to midi matrix """

    n_frames = int(np.ceil((notes[-1, 0] + notes[-1, 2]) / dt))
    midi_matrix = np.zeros((128, n_frames), dtype=np.uint8)
    for n in notes:
        onset = int(np.ceil(n[0] / dt))
        offset = int(np.ceil((n[0] + n[2]) / dt))
        midi_pitch = int(n[1])
        midi_matrix[midi_pitch, onset:offset] += 1

    return midi_matrix


class MidiParser(object):
    """
    Compute spectrogram from audio and parse note onsets from midi file
    """

    def __init__(self, show=False):
        """
        Constructor
        """
        self.show = show

    def process(self, midi_file_path, audio_path=None, return_midi_matrix=False):
        """
        Process midi file
        """

        # compute spectrogram
        Spec = None
        if audio_path is not None:
            logging.info('Computing spectrogram from audio path: {0}'.format(audio_path))
            if not os.path.isfile(audio_path):
                logging.info('...audio file does not exist!')

            Spec = processor(audio_path).T

        # show results
        if self.show and Spec is not None:
            plt.figure('spec')
            plt.subplot(111)
            plt.subplots_adjust(top=1.0, bottom=0.0)
            plt.imshow(Spec, cmap='viridis', interpolation='nearest', aspect='auto', origin='lower')
            plt.colorbar()

        s = music21.midi.translate.midiFilePathToStream(midi_file_path)

        mf = music21.midi.translate.streamToMidiFile(s)
        n_tracks = len(mf.tracks)

        for i_track, track in enumerate(mf.tracks[0:1]):
            #            st = music21.midi.translate.midiTrackToStream(track, ticksPerQuarter=mf.ticksPerQuarterNote)
            #
            #            time, onsets = 0.0, []
            #            for i_evt, evt in enumerate(track.events):
            #
            #                if evt.isDeltaTime() and evt.time > 0:
            #                    onsets.append(time / 1024 * 60)
            #                    time += evt.time
            #
            #                if evt.isNoteOn():
            #                    print evt
            #
            #            # write temporary midi
            #            mfile = "tmp/tmp_%d.mid" % i_track
            #            st.write('midi', mfile)

            # parse midi file
            # m = mm_midi.MIDIFile.from_file(mfile)
            m = mm_midi.MIDIFile.from_file(midi_file_path)
            # print midi_file_path, m.notes()

            # Order notes by onset and top-down in simultaneities
            notes = np.asarray(sorted(m.notes(), key=lambda n: (n[0], n[1] * -1)))
            onsets = notes_to_onsets(notes, dt=1.0 / FPS)
            midi_matrix = notes_to_matrix(notes, dt=1.0 / FPS)

            # show results
            if self.show:
                # print onsets
                for o in onsets:
                    plt.plot([o, o], [0, Spec.shape[0] - 1], '-', color=colors[i_track], linewidth=2, alpha=1.0)
                # plt.xlim([0, Spec.shape[1]-1])
                plt.ylim([0, Spec.shape[0] - 1])

        if self.show:
            plt.show(block=True)

        if return_midi_matrix:
            return Spec, onsets, midi_matrix, notes
        else:
            return Spec, onsets


if __name__ == '__main__':
    """
    main
    """

    # midi file
    pattern = "/home/matthias/cp/data/sheet_localization/real_music/Mozart_Piano_Sonata_No_16_Allegro/audio/*.midi"
    for midi_file_path in glob.glob(pattern):
        print midi_file_path

        # get file names and directories
        directory = os.path.dirname(midi_file_path)
        file_name = os.path.basename(midi_file_path).split('.midi')[0]
        audio_file_path = os.path.join(directory, file_name + '.flac')
        spec_file_path = os.path.join(directory.replace("/audio", "/spec"), file_name + '_spec.npy')
        onset_file_path = os.path.join(directory.replace("/audio", "/spec"), file_name + '_onsets.npy')

        # parse midi file
        midi_parser = MidiParser(show=True)
        Spec, onsets = midi_parser.process(midi_file_path, audio_file_path)

        np.save(spec_file_path, Spec)
        np.save(onset_file_path, onsets)
