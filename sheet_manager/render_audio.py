# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 18:49:33 2016

@author: matthias
"""

import os
import midi as midipy
from mido import MidiFile, MidiTrack, bpm2tempo, tempo2bpm


def set_tempo_events(filename_in, filename_out, tempoevents):
    """
    set midi tempo events
    """
    midifile = midipy.read_midifile(filename_in)
    midifile.make_ticks_abs()

    max_tick = 0

    for track in midifile:
        for event in track:
            if not isinstance(event, midipy.EndOfTrackEvent):
                max_tick = max(event.tick, max_tick)
            if isinstance(event, midipy.SetTempoEvent):
                track.remove(event)

    for track in midifile:
        for event in track:
            if isinstance(event, midipy.EndOfTrackEvent):
                event.tick = max_tick + 1

    # TODO: adapt tempo for changing signatures
    # time_signatures = get_time_signature_events(filename_in)
    #
    # adaptions = []
    #
    # for ts in time_signatures:
    #     add = [ts.tick, 0, 4.0 / ts.denom]
    #     adaptions.append(add)
    #
    # adaptions[-1][1] = tempoevents[-1].tick + 1
    #
    # for row in adaptions:
    #     for e in tempoevents:
    #         if e.tick >= row[0] and e.tick < row[1]:
    #             e.bpm = int(e.bpm * row[2])

    for e in tempoevents:
        te = midipy.SetTempoEvent(tick=e.tick, bpm=e.bpm)
        midifile[0].append(te)

    for track in midifile:
        track.sort()

    midifile.make_ticks_rel()
    midipy.write_midifile(filename_out, midifile)


def set_velocity(filename_in, filename_out, velocity):
    """
    set velocities
    """
    midifile = midipy.read_midifile(filename_in)

    for track in midifile:
        for event in track:
            if isinstance(event, midipy.NoteOnEvent):
                event.set_velocity(velocity)

    midipy.write_midifile(filename_out, midifile)


class TempoEvent:
    """
    Helper class for tempo events
    """

    def __init__(self, tick, bpm):
        self.tick = tick
        self.bpm = bpm


def change_midi_file_tempo(input_file, output_file, ratio=1.0):
    """
    Change the tempo in a midi file
    """
    infile = MidiFile(input_file)

    with MidiFile() as outfile:

        outfile.type = infile.type
        outfile.ticks_per_beat = infile.ticks_per_beat

        for msg_idx, track in enumerate(infile.tracks):

            out_track = MidiTrack()
            outfile.tracks.append(out_track)

            for message in track:
                if message.type == 'set_tempo':
                    bpm = tempo2bpm(message.tempo)
                    new_bpm = ratio * bpm
                    message.tempo = bpm2tempo(new_bpm)

                out_track.append(message)

    outfile.save(output_file)


def render_audio(input_midi_path, sound_font,
                 velocity=None,
                 change_tempo=True, tempo_ratio=None,
                 target_dir=None,
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

    audio_directory = "tmp" # os.path.join(directory, 'audio')
    if not os.path.exists(audio_directory):
        os.mkdir(audio_directory)

    # Generate the output filename base string
    new_file_name, midi_extension = os.path.splitext(file_name)

    if tempo_ratio:
        new_file_name += "_temp_%d" % int(1000 * tempo_ratio)

    if velocity:
        new_file_name += "_vel%d" % velocity

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

    # try:
    #     from midi2audio import FluidSynth
    #     fs = FluidSynth(sound_font_path)
    #     fs.midi_to_audio(filename_out, audio_file=audio_file)
    # except ImportError:
    # synthesize midi file to audio
    cmd = "fluidsynth -F %s -O s16 -T flac %s %s" % (audio_path,
                                                     sound_font_path,
                                                     perf_midi_path)
    os.system(cmd)

    return audio_path, perf_midi_path


##############################################################################


if __name__ == '__main__':
    """
    main
    """

    sound_fonts = ["Acoustic_Piano", "Equinox_Grand_Pianos", "FluidR3_GM",
                   "Steinway", "Unison"]

    sound_fonts = ["Acoustic_Piano"]

    # select tempo
    for bpm in [120]:

        # set velocity
        for velocity in [120]:

            # select sound fount
            for sound_font in sound_fonts:
                # path to sound font
                sound_font_path = "/home/matthias/cp/data/soundfonts/%s.sf2" % sound_font

                # the music xml file
                filename_in = "/home/matthias/cp/data/sheet_localization/real_music/mozart_piano_sonata_no_16_allegro/Mozart_Piano_Sonata_No_16_Allegro.midi"

                # get file names and directories
                file_name = os.path.basename(filename_in)
                directory = os.path.dirname(filename_in)
                audio_directory = os.path.join(directory, 'audio')

                new_file_name = file_name.split('.midi')[0] + "_vel%d_%dbpm_%s" % (velocity, bpm, sound_font)
                filename_out = os.path.join(audio_directory, new_file_name + ".midi")
                audio_file = os.path.join(audio_directory, new_file_name + ".flac")

                # fix midi tempo
                set_tempo_events(filename_in, filename_out, [TempoEvent(0, bpm)])

                # set velocity
                set_velocity(filename_out, filename_out, velocity)

                # synthetizes midi file to audio
                os.system("fluidsynth -F %s -O s16 -T flac %s %s" % (audio_file, sound_font_path, filename_out))
