# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:02:37 2016

@author: matthias

Synthesize audio from midi and extract spectrogram annotations

"""
from __future__ import print_function

import logging
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import madmom.io.midi as mm_midi
import pretty_midi
# import madmom.utils.midi_old as mm_midi
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.filters import LogarithmicFilterbank
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor
from madmom.processors import SequentialProcessor


class MidiParser(object):
    """
    Compute spectrogram from audio and parse note onsets from midi file
    """

    def __init__(self, show=False):
        """
        Constructor
        """
        self.show = show

    def process(self, midi_file_path, audio_path=None, return_midi_matrix=False,
                frame_size=2048, sample_rate=22050, fps=20, num_bands=16, fmin=30, fmax=6000):
        """
        Process midi file
        """

        # compute spectrogram
        spec = None
        if audio_path is not None:
            logging.info('Computing spectrogram from audio path: {0}'.format(audio_path))
            if not os.path.isfile(audio_path):
                logging.info('...audio file does not exist!')

            spec = extract_spectrogram(audio_path, frame_size=frame_size, sample_rate=sample_rate, fps=fps,
                                       num_bands=num_bands, fmin=fmin, fmax=fmax)

        # show results
        if self.show and spec is not None:
            plt.figure('spec')
            plt.subplot(111)
            plt.subplots_adjust(top=1.0, bottom=0.0)
            plt.imshow(spec, cmap='viridis', interpolation='nearest', aspect='auto', origin='lower')
            plt.colorbar()

        # load midi file
        m = mm_midi.MIDIFile(midi_file_path)

        # Order notes by onset and top-down in simultaneities
        notes = np.asarray(sorted(m.notes, key=lambda n: (n[0], n[1] * -1)))
        onsets = notes_to_onsets(notes, dt=1.0 / fps)
        durations = np.asarray([int(np.ceil(n[2] * fps)) for n in notes])
        midi_matrix = notes_to_matrix(notes, dt=1.0 / fps)

        if self.show:
            plt.show(block=True)

        if return_midi_matrix:
            return spec, onsets, durations, midi_matrix, notes
        else:
            return spec, onsets, durations


def extract_spectrogram(audio_path, frame_size=2048, sample_rate=22050, fps=20,
                        num_bands=16, fmin=30, fmax=6000):
    sig_proc = SignalProcessor(num_channels=1, sample_rate=sample_rate)
    fsig_proc = FramedSignalProcessor(frame_size=frame_size, fps=fps, origin='future')
    spec_proc = FilteredSpectrogramProcessor(LogarithmicFilterbank, num_bands=num_bands, fmin=fmin, fmax=fmax)
    log_spec_proc = LogarithmicSpectrogramProcessor()
    processor = SequentialProcessor([sig_proc, fsig_proc, spec_proc, log_spec_proc])

    return processor(audio_path).T


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


def change_midi_file_velocity(filename_in, filename_out, velocity):
    """
    set velocities
    """

    midi_data = pretty_midi.PrettyMIDI(filename_in)

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            note.velocity = int(velocity)

    midi_data.write(filename_out)


def change_midi_file_program(filename_in, filename_out, program):
    """
    set instruments
    """

    midi_data = pretty_midi.PrettyMIDI(filename_in)

    for instrument in midi_data.instruments:
        instrument.program = program

    midi_data.write(filename=filename_out)


def change_midi_file_tempo(input_file, output_file, ratio=1.0):
    """
    Change the tempo in a midi file in a relative way.
    """

    midi_data = pretty_midi.PrettyMIDI(input_file)

    # infile = MidiFile(input_file)
    new_tempi = []

    for n, (tick, tick_scale) in enumerate(midi_data._tick_scales):
        # Convert tick of this tempo change to time in seconds
        tempo_event = (tick, tick_scale*ratio)
        new_tempi.append(tempo_event)

    # set new tempo to midi
    midi_data._tick_scales = new_tempi

    midi_data.write(filename=output_file)


def render_audio(input_midi_path, sound_font,
                 velocity=None,
                 change_tempo=True, tempo_ratio=None,
                 target_dir=None,
                 quiet=True,
                 audio_fmt=".flac",
                 sound_font_root="~/.fluidsynth"):
    """
    Render midi to audio.

    Returns the output audio and midi filenames.
    """
    # path to sound font
    sound_font_path = os.path.join(sound_font_root, "%s.sf2" % sound_font)

    # get file names and directories
    file_name = os.path.basename(input_midi_path)
    directory = target_dir if target_dir else os.path.dirname(input_midi_path)

    audio_directory = "tmp"  # os.path.join(directory, 'audio')
    if not os.path.exists(audio_directory):
        os.mkdir(audio_directory)

    # Generate the output filename base string
    new_file_name, midi_extension = os.path.splitext(file_name)

    if tempo_ratio:
        new_file_name += "_temp_%d" % int(1000 * tempo_ratio)

    # if velocity:
    #     new_file_name += "_vel%d" % velocity

    new_file_name += "_%s" % sound_font

    # set file names
    if audio_fmt.startswith("."):
        audio_fmt = audio_fmt[1:]
    audio_fname = new_file_name + "." + audio_fmt
    audio_path = os.path.join(audio_directory, audio_fname)

    perf_midi_fname = new_file_name + "." + midi_extension
    perf_midi_path = os.path.join(audio_directory, perf_midi_fname)

    # prepare midi
    if change_tempo:
        change_midi_file_tempo(input_midi_path, perf_midi_path, tempo_ratio)

    if velocity:
        change_midi_file_velocity(perf_midi_path, perf_midi_path, velocity)

    # synthesize MIDI with fluidsynth
    cmd = "fluidsynth -F %s -O s16 -T flac %s %s" % (audio_path,
                                                     sound_font_path,
                                                     perf_midi_path)
    if quiet:
        cmd += ' > /dev/null'

    os.system(cmd)

    return audio_path, perf_midi_path


if __name__ == '__main__':
    """
    main
    """

    # midi file
    pattern = "/home/matthias/cp/data/sheet_localization/real_music/Mozart_Piano_Sonata_No_16_Allegro/audio/*.midi"
    for midi_file_path in glob.glob(pattern):
        print(midi_file_path)

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
