
# Soundfonts

These are the soundfonts used to render the midi files.
We use different soundfonts for training and testing to see weather the models generalize.

## Train soundfonts
- FluidR3_GM (here we use the two presents ElectricPiano, YamahaGrandPiano)
- acoustic_piano_imis_1.sf2 (http://zenvoid.org/audio/)

## Test soundfont
- grand-piano-YDP-20160804.tar.bz2 (https://musescore.org/de/handbook/soundfonts-und-sfz-dateien)


# Alignments


Note that we only align the default score to one performance. However, because
the performances are synthesized, this does not matter: they contain
exactly the same note events, with exactly the same simultaneities,
just the absolute timing changes different (because of tempo augmentations).

(This stops being fine with live MIDI that has mistakes.)