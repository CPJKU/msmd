"""This file contains the default Sheet Manager configuration values
and their documentation. Use it to seed other configuration files."""
import os

import numpy

try:
    from LOCAL_CONFIG import *
except ImportError:

    # Data root
    DATA_ROOT_MSMD = '/Users/hajicj/data/MSMD/msmd_aug'
    # get hostname
    hostname = os.uname()[1]

    # adopted paths
    if hostname in ["rechenknecht0.cp.jku.at", "rechenknecht1.cp.jku.at"]:
        DATA_ROOT_MSMD = '/home/matthias/shared/datasets/msmd_aug/'

    elif hostname == "mdhp":
        DATA_ROOT_MSMD = '/media/matthias/Data/Data/msmd/'

    elif hostname.endswith('ufal.mff.cuni.cz'):
        DATA_ROOT_MSMD = '/lnet/data/msmd'

    # PDF --> PNG rendering
    # ---------------------

    TARGET_WIDTH = 835
    #: Width of the rendered PNG pages in pixels

    PNG_DENSITY = 150
    #: Resolution in DPI of the output PNG

    PNG_QUALITY = 90
    #: Quality of images rendered from PDF.

    # Performance rendering
    # ---------------------

    tempo_ratios = numpy.arange(0.9, 1.1, 0.025)  # [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
    #: The performance MIDIs will be rendered at all these tempo ratios.

    sound_font_root = "~/.fluidsynth"
    #: The directory containing the soundfonts for performance audio rendering

    sound_fonts = ["Acoustic_Piano", "Unison", "FluidR3_GM", "Steinway"]
    #: A performance MIDIs will be rendered for each of these soundfonts.

    # Features
    # --------

    SIGNAL_NUM_CHANNELS = 1
    #: The input audio signal will be reduced to this number of channels.

    FPS = 20
    #: Frames per second; translates to no. of feature matrix columns per second
    #  of audio. Frames are draws from the audio with this frequency.
    #  (This is an extremely important setting!)

    SAMPLE_RATE = 22050
    #: Use this sample rate when processing performance audio.

    FRAME_SIZE = 2048
    #: How many samples will be aggregated into one frame.

    # MIDI_MATRIX_PITCH_BINS = 128
    #: All MIDI has 128 pitches.

    # Log filterbank for spectrogram computation
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    SPEC_FILTERBANK_NUM_BANDS = 16
    #: Number of frequency bands per octave in the log filterbank.

    FMIN = 30
    #: Lowest filter frequency.

    FMAX = 6000
    #: Highest filter frequency.

    # OMR modules
    # -----------

    OMR_RELATIVE_DIR = 'omr_models'
    #: Path of the OMR model params dir relative to the sheet_manager.py file.

    # OMR_ABSOLUTE_DIR = None
    #: Absolute path of the OMR models dir. Overrides OMR_RELATIVE_DIR;
    #  use this if you have your own OMR models and don't want to use
    #  the pre-trained models that Sheet Manager provides. Note that
    #  the architectures must still conform to the SegmentationNetwork;
    #  only the parameters are stored in these directories.

