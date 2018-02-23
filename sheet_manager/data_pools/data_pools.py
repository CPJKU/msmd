
from __future__ import print_function

import logging

import numpy as np
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt


from sheet_manager.midi_parser import notes_to_onsets, FPS

from sheet_manager.data_model.piece import Piece
from sheet_manager.alignments import align_score_to_performance


# Note that sheet context is applied to *both* sides.
# SHEET_CONTEXT = 200
SHEET_CONTEXT_LEFT = 100
SHEET_CONTEXT_RIGHT = 20
SHEET_CONTEXT = SHEET_CONTEXT_LEFT + SHEET_CONTEXT_RIGHT

SYSTEM_HEIGHT = 160

SPEC_CONTEXT = 42
SPEC_BINS = 92

NO_AUGMENT = dict()
NO_AUGMENT['system_translation'] = 0
NO_AUGMENT['sheet_scaling'] = [1.00, 1.00]
NO_AUGMENT['onset_translation'] = 0
NO_AUGMENT['spec_padding'] = 0
NO_AUGMENT['interpolate'] = -1
NO_AUGMENT['synths'] = ['ElectricPiano']
NO_AUGMENT['tempo_range'] = [1.00, 1.00]

# this will be overwritten with a config file
# (see audio_sheet_retrieval/exp_configs)
AUGMENT = dict()
for key in NO_AUGMENT.keys():
    AUGMENT[key] = NO_AUGMENT[key]


class AudioScoreRetrievalPool(object):

    def __init__(self, images, specs, o2c_maps,
                 spec_context=SPEC_CONTEXT, sheet_context=SHEET_CONTEXT, staff_height=SYSTEM_HEIGHT,
                 data_augmentation=None, shuffle=True):

        self.images = images
        self.specs = specs
        self.o2c_maps = o2c_maps

        self.spec_context = spec_context
        self.sheet_context = sheet_context
        self.staff_height = staff_height

        self.data_augmentation = data_augmentation
        self.shuffle = shuffle

        self.shape = None
        self.sheet_dim = [self.staff_height, self.sheet_context]
        self.spec_dim = [self.specs[0][0].shape[0], self.spec_context]

        if self.data_augmentation['interpolate'] > 0:
            self.interpolate()

        self.prepare_train_entities()

        if self.shuffle:
            self.reset_batch_generator()

    def interpolate(self):
        """
        Interpolate onset to note correspondences on frame level
        """
        for i_sheet in range(len(self.images)):
            for i_spec in range(len(self.specs[i_sheet])):

                onsets = self.o2c_maps[i_sheet][i_spec][:, 0]
                coords = self.o2c_maps[i_sheet][i_spec][:, 1]

                # interpolate some extra onsets
                step_size = self.data_augmentation['interpolate']
                f_inter = interp1d(onsets, coords)
                onsets = np.arange(onsets[0], onsets[-1] + 1, step_size)
                coords = f_inter(onsets)

                # update mapping
                onsets = onsets.reshape((-1, 1))
                coords = coords.reshape((-1, 1))
                new_mapping = np.hstack((onsets, coords))
                self.o2c_maps[i_sheet][i_spec] = new_mapping.astype(np.int)

    def prepare_train_entities(self):
        """
        Collect train entities
        """

        self.train_entities = np.zeros((0, 3), dtype=np.int)

        # iterate sheets
        for i_sheet, sheet in enumerate(self.images):

            # iterate spectrograms
            for i_spec, spec in enumerate(self.specs[i_sheet]):

                # iterate onsets in sheet
                for i_onset in range(len(self.o2c_maps[i_sheet][i_spec])):

                    onset = self.o2c_maps[i_sheet][i_spec][i_onset, 0]
                    o_start = onset - self.spec_context // 2
                    o_stop = o_start + self.spec_context

                    coord = self.o2c_maps[i_sheet][i_spec][i_onset, 1]
                    c_start = coord - self.sheet_context // 2
                    c_stop = o_start + self.sheet_context

                    if o_start >= 0 and o_stop < spec.shape[1]\
                            and c_start >= 0 and c_stop < sheet.shape[1]:
                        cur_entities = np.asarray([i_sheet, i_spec, i_onset])
                        self.train_entities = np.vstack((self.train_entities, cur_entities))

        # number of train samples
        self.shape = [self.train_entities.shape[0]]

    def reset_batch_generator(self):
        """
        Reset data pool
        """
        indices = np.random.permutation(self.shape[0])
        self.train_entities = self.train_entities[indices]

    def prepare_train_image(self, i_sheet, i_spec, i_onset, shift=None):
        """ prepare train item """

        # get sheet and annotations
        sheet = self.images[i_sheet]

        # get target note coodinate
        target_coord = self.o2c_maps[i_sheet][i_spec][i_onset][1]

        # get sub-image (with coordinate fixing)
        c0 = max(0, target_coord - 2 * self.sheet_context)
        c1 = min(c0 + 4 * self.sheet_context, sheet.shape[1])
        c0 = max(0, c1 - 4 * self.sheet_context)
        sheet = sheet[:, c0:c1]

        if self.data_augmentation['sheet_scaling']:
            import cv2
            sc = self.data_augmentation['sheet_scaling']
            scale = (sc[1] - sc[0]) * np.random.random_sample() + sc[0]
            new_size = (int(sheet.shape[1] * scale), int(sheet.shape[0] * scale))
            sheet = cv2.resize(sheet, new_size, interpolation=cv2.INTER_NEAREST)

        # target coordinate
        x = sheet.shape[1] // 2

        # compute sliding window coordinates
        x0 = np.max([x - self.sheet_context // 2, 0])
        x1 = x0 + self.sheet_context

        x1 = int(np.min([x1, sheet.shape[1] - 1]))
        x0 = int(x1 - self.sheet_context)

        # get vertical crop
        r0 = sheet.shape[0] // 2 - self.staff_height // 2
        if self.data_augmentation['system_translation']:
            t = self.data_augmentation['system_translation']
            r0 += np.random.randint(low=-t, high=t + 1)
        r1 = r0 + self.staff_height

        # get sheet snippet
        sheet_snippet = sheet[r0:r1, x0:x1]

        return sheet_snippet

    def prepare_train_audio(self, i_sheet, i_spec, i_onset):
        """
        Prepare audio excerpt
        """

        # get spectrogram and onset
        spec = self.specs[i_sheet][i_spec]
        sel_onset = int(self.o2c_maps[i_sheet][i_spec][i_onset][0])

        # data augmentation note position
        if self.data_augmentation['onset_translation']:
            t = self.data_augmentation['onset_translation']
            sel_onset += np.random.randint(low=-t, high=t + 1)

        # compute sliding window coordinates
        start = np.max([sel_onset - self.spec_context // 2, 0])
        stop = start + self.spec_context

        stop = np.min([stop, spec.shape[1] - 1])
        start = stop - self.spec_context

        excerpt = spec[:, start:stop]

        if self.data_augmentation['spec_padding']:
            spec_padding = self.data_augmentation['spec_padding']
            excerpt = np.pad(excerpt, ((spec_padding, spec_padding), (0, 0)), mode='edge')
            s = np.random.randint(0, spec_padding)
            e = s + spec.shape[0]
            excerpt = excerpt[s:e, :]

        return excerpt

    def __getitem__(self, key):
        """
        Make class accessible by index or slice
        """

        # get batch
        if key.__class__ == int:
            key = slice(key, key + 1)
        batch_entities = self.train_entities[key]

        # collect train entities
        sheet_batch = np.zeros((len(batch_entities), 1, self.sheet_dim[0], self.sheet_context), dtype=np.float32)
        spec_batch = np.zeros((len(batch_entities), 1, self.spec_dim[0], self.spec_context), dtype=np.float32)
        for i_entity, (i_sheet, i_spec, i_onset) in enumerate(batch_entities):

            # get sliding window train item
            snippet = self.prepare_train_image(i_sheet, i_spec, i_onset)

            # get spectrogram excerpt (target note in center)
            excerpt = self.prepare_train_audio(i_sheet, i_spec, i_onset)

            # collect batch data
            sheet_batch[i_entity, 0, :, :] = snippet
            spec_batch[i_entity, 0, :, :] = excerpt

        return [sheet_batch, spec_batch]


class ScoreInformedTranscriptionPool(object):

    def __init__(self, images, specs, o2c_maps, midi_matrices,
                 spec_context=SPEC_CONTEXT,
                 sheet_context_left=SHEET_CONTEXT_LEFT,
                 sheet_context_right=SHEET_CONTEXT_RIGHT,
                 staff_height=SYSTEM_HEIGHT,
                 data_augmentation=None, shuffle=True):

        self.images = images
        self.specs = specs
        self.o2c_maps = o2c_maps
        self.midi_matrices = midi_matrices

        self.spec_context = spec_context

        self.sheet_context_left = sheet_context_left
        self.sheet_context_right = sheet_context_right

        self.sheet_context = sheet_context_left + sheet_context_right
        self.staff_height = staff_height

        self.data_augmentation = data_augmentation
        self.shuffle = shuffle

        self.shape = None
        self.sheet_dim = [self.staff_height, self.sheet_context]
        self.spec_dim = [self.specs[0][0].shape[0], self.spec_context]

        if self.data_augmentation['interpolate'] > 0:
            self.interpolate()

        self.prepare_train_entities()

        if self.shuffle:
            self.reset_batch_generator()

    def interpolate(self):
        """
        Interpolate onset to note correspondences on frame level
        """
        for i_sheet in range(len(self.images)):
            for i_spec in range(len(self.specs[i_sheet])):

                onsets = self.o2c_maps[i_sheet][i_spec][:, 0]
                coords = self.o2c_maps[i_sheet][i_spec][:, 1]

                # interpolate some extra onsets
                step_size = self.data_augmentation['interpolate']
                f_inter = interp1d(onsets, coords)
                onsets = np.arange(onsets[0], onsets[-1] + 1, step_size)
                coords = f_inter(onsets)

                # update mapping
                onsets = onsets.reshape((-1, 1))
                coords = coords.reshape((-1, 1))
                new_mapping = np.hstack((onsets, coords))
                self.o2c_maps[i_sheet][i_spec] = new_mapping.astype(np.int)

    def prepare_train_entities(self):
        """
        Collect train entities
        """

        self.train_entities = np.zeros((0, 3), dtype=np.int)

        # iterate sheets
        for i_sheet, sheet in enumerate(self.images):

            # iterate spectrograms
            for i_spec, spec in enumerate(self.specs[i_sheet]):

                # iterate onsets in sheet
                for i_onset in range(len(self.o2c_maps[i_sheet][i_spec])):

                    onset = self.o2c_maps[i_sheet][i_spec][i_onset, 0]
                    o_start = onset - self.spec_context // 2
                    o_stop = o_start + self.spec_context

                    coord = self.o2c_maps[i_sheet][i_spec][i_onset, 1]
                    c_start = coord - self.sheet_context // 2
                    c_stop = o_start + self.sheet_context

                    if o_start >= 0 and o_stop < spec.shape[1]\
                            and c_start >= 0 and c_stop < sheet.shape[1]:
                        cur_entities = np.asarray([i_sheet, i_spec, i_onset])
                        self.train_entities = np.vstack((self.train_entities, cur_entities))

        # number of train samples
        self.shape = [self.train_entities.shape[0]]

    def reset_batch_generator(self):
        """
        Reset data pool
        """
        indices = np.random.permutation(self.shape[0])
        self.train_entities = self.train_entities[indices]

    def prepare_train_image(self, i_sheet, i_spec, i_onset, shift=None):
        """ prepare train item """

        # get sheet and annotations
        sheet = self.images[i_sheet]

        # get target note coodinate
        target_coord = self.o2c_maps[i_sheet][i_spec][i_onset][1]

        # get sub-image (with coordinate fixing)
        c0 = max(0, target_coord - 2 * self.sheet_context_left)
        c1 = min(c0 + 2 * self.sheet_context_left + 2 * self.sheet_context_right, sheet.shape[1])
        c0 = max(0, c1 - 4 * self.sheet_context)
        sheet = sheet[:, c0:c1]

        if self.data_augmentation['sheet_scaling']:
            import cv2
            sc = self.data_augmentation['sheet_scaling']
            scale = (sc[1] - sc[0]) * np.random.random_sample() + sc[0]
            new_size = (int(sheet.shape[1] * scale), int(sheet.shape[0] * scale))
            sheet = cv2.resize(sheet, new_size, interpolation=cv2.INTER_NEAREST)

        # target coordinate
        # ...due to scaling, we have to base this on the ratio between
        #    between the sheet left context and right context.
        x = int(sheet.shape[1] * (self.sheet_context_left / self.sheet_context))

        # compute sliding window coordinates
        x0 = np.max([x - self.sheet_context_left, 0])
        x1 = x0 + self.sheet_context

        x1 = int(np.min([x1, sheet.shape[1] - 1]))
        x0 = int(x1 - self.sheet_context)

        # get vertical crop
        r0 = sheet.shape[0] // 2 - self.staff_height // 2
        if self.data_augmentation['system_translation']:
            t = self.data_augmentation['system_translation']
            r0 += np.random.randint(low=-t, high=t + 1)
        r1 = r0 + self.staff_height

        # get sheet snippet
        sheet_snippet = sheet[r0:r1, x0:x1]

        return sheet_snippet

    def prepare_train_audio(self, i_sheet, i_spec, i_onset):
        """
        Prepare audio excerpt
        """

        # get spectrogram and onset
        spec = self.specs[i_sheet][i_spec]
        sel_onset = int(self.o2c_maps[i_sheet][i_spec][i_onset][0])

        # get midi matrix
        midi_matrix = self.midi_matrices[i_sheet][i_spec]

        # data augmentation note position
        if self.data_augmentation['onset_translation']:
            t = self.data_augmentation['onset_translation']
            sel_onset += np.random.randint(low=-t, high=t + 1)

        # compute sliding window coordinates
        start = np.max([sel_onset - self.spec_context // 2, 0])
        stop = start + self.spec_context

        stop = np.min([stop, spec.shape[1] - 1])
        start = stop - self.spec_context

        # get spectrogram and midi matrix excerpt
        excerpt = spec[:, start:stop]
        midi_excerpt = midi_matrix[:, start:stop]

        if midi_excerpt.shape[-1] != self.spec_context:
            raise ValueError('MIDI excerpt is not large enough for the spec. context! Excerpt shape: {0},'
                             ' midi matrix shape: {1}, spectrogram shape: {2}, idx: {3}, start: {4}, stop: {5}'
                             ''.format(midi_excerpt.shape, midi_matrix.shape, spec.shape, i_sheet, start, stop))

        if self.data_augmentation['spec_padding']:
            spec_padding = self.data_augmentation['spec_padding']
            excerpt = np.pad(excerpt, ((spec_padding, spec_padding), (0, 0)), mode='edge')
            s = np.random.randint(0, spec_padding)
            e = s + spec.shape[0]
            excerpt = excerpt[s:e, :]

        return excerpt, midi_excerpt

    def __getitem__(self, key):
        """
        Make class accessible by index or slice
        """

        # get batch
        if key.__class__ == int:
            key = slice(key, key + 1)
        batch_entities = self.train_entities[key]

        # collect train entities
        sheet_batch = np.zeros((len(batch_entities), 1, self.sheet_dim[0], self.sheet_context), dtype=np.float32)
        spec_batch = np.zeros((len(batch_entities), 1, self.spec_dim[0], self.spec_context), dtype=np.float32)
        midi_batch = np.zeros((len(batch_entities), 1, 128, self.spec_context), dtype=np.float32)
        for i_entity, (i_sheet, i_spec, i_onset) in enumerate(batch_entities):

            # get sliding window train item
            snippet = self.prepare_train_image(i_sheet, i_spec, i_onset)

            # get spectrogram excerpt (target note in center)
            excerpt, midi_excerpt = self.prepare_train_audio(i_sheet, i_spec, i_onset)

            if midi_excerpt.shape != midi_batch.shape[-2:]:
                raise ValueError('Wrong shape of MIDI excerpt: key {0}'.format(key))

            # collect batch data
            sheet_batch[i_entity, 0, :, :] = snippet
            spec_batch[i_entity, 0, :, :] = excerpt
            midi_batch[i_entity, 0, :, :] = midi_excerpt

        return [sheet_batch, spec_batch, midi_batch]


def onset_to_coordinates(alignment, mdict, note_events):
    """
    Compute onset to coordinate mapping
    """
    onset_to_coord = np.zeros((0, 2), dtype=np.int)

    for m_objid, e_idx in alignment:

        # get note mungo and midi note event
        m, e = mdict[m_objid], note_events[e_idx]

        # compute onset frame
        onset_frame = notes_to_onsets([e], dt=1.0 / FPS)

        # get note coodinates
        cy, cx = m.middle

        # keep mapping
        entry = np.asarray([onset_frame, cx], dtype=np.int)[np.newaxis]
        if onset_frame not in onset_to_coord[:, 0]:
            onset_to_coord = np.concatenate((onset_to_coord, entry), axis=0)

    return onset_to_coord


def systems_to_rois(sys_mungos, window_top=10, window_bottom=10):
    """
    Convert systems to rois
    """

    page_rois = np.zeros((0, 4, 2))
    for sys_mungo in sys_mungos:
        t, l, b, r = sys_mungo.bounding_box

        cr = (t + b) // 2

        r_min = cr - window_top
        r_max = r_min + window_top + window_bottom
        c_min = l
        c_max = r

        topLeft = [r_min, c_min]
        topRight = [r_min, c_max]
        bottomLeft = [r_max, c_min]
        bottomRight = [r_max, c_max]
        system = np.asarray([topLeft, topRight, bottomRight, bottomLeft])
        system = system.reshape((1, 4, 2))
        page_rois = np.vstack((page_rois, system))

    return page_rois.astype(np.int)


def stack_images(images, mungos_per_page, mdict):
    """
    Re-stitch image
    """
    stacked_image = images[0]
    stacked_page_mungos = [m for m in mungos_per_page[0]]

    row_offset = stacked_image.shape[0]

    for i in range(1, len(images)):

        # append image
        stacked_image = np.concatenate((stacked_image, images[i]))

        # update coordinates
        page_mungos = mungos_per_page[i]
        for m in page_mungos:
            m.x += row_offset
            stacked_page_mungos.append(m)
            mdict[m.objid] = m

        # update row offset
        row_offset = stacked_image.shape[0]

    return stacked_image, stacked_page_mungos, mdict


def unwrap_sheet_image(image, system_mungos, mdict, window_top=100, window_bottom=100):
    """
    Unwrap all systems of sheet image to a single "long row"
    """

    # get rois from page systems
    rois = systems_to_rois(system_mungos, window_top, window_bottom)

    width = image.shape[1] * rois.shape[0]
    window = rois[0, 3, 0] - rois[0, 0, 0]

    un_wrapped_coords = dict()
    un_wrapped_image = np.zeros((window, width), dtype=np.uint8)

    # make single staff image
    x_offset = 0
    img_start = 0
    for j, sys_mungo in enumerate(system_mungos):

        # get current roi
        r = rois[j]

        # fix out of image errors
        pad_top = 0
        pad_bottom = 0
        if r[0, 0] < 0:
            pad_top = np.abs(r[0, 0])
            r[0, 0] = 0

        if r[3, 0] >= image.shape[0]:
            pad_bottom = r[3, 0] - image.shape[0]

        # get system image
        system_image = image[r[0, 0]:r[3, 0], r[0, 1]:r[1, 1]]

        # pad missing rows and fix coordinates
        system_image = np.pad(system_image, ((pad_top, pad_bottom), (0, 0)), mode='edge')

        img_end = img_start + system_image.shape[1]
        un_wrapped_image[:, img_start:img_end] = system_image

        # get noteheads of current staff
        staff_noteheads = [mdict[i] for i in sys_mungo.inlinks if mdict[i].clsname == 'notehead-full']

        # compute unwraped coordinates
        for n in staff_noteheads:
            n.x -= r[0, 0]
            n.y += x_offset - r[0, 1]
            un_wrapped_coords[n.objid] = n

        x_offset += (r[1, 1] - r[0, 1])
        img_start = img_end

    # get relevant part of unwrapped image
    un_wrapped_image = un_wrapped_image[:, :img_end]

    return un_wrapped_image, un_wrapped_coords


def prepare_piece_data(collection_dir, piece_name, aug_config=NO_AUGMENT, require_audio=True, load_midi_matrix=False):
    """
    :param collection_dir:
    :param piece_name:
    :return:
    """

    logging.info("\n")
    logging.info("{0}:\tPiece loading".format(piece_name))
    piece = Piece(root=collection_dir, name=piece_name)
    logging.info("{0}:\tScore loading".format(piece_name))
    score = piece.load_score(piece.available_scores[0])

    logging.info("{0}:\tMuNGo loading".format(piece_name))
    mungos = score.load_mungos()
    mdict = {m.objid: m for m in mungos}
    mungos_per_page = score.load_mungos(by_page=True)

    logging.info("{0}:\tImage loading".format(piece_name))
    images = score.load_images()

    # stack sheet images
    logging.info("{0}:\tImage stacking".format(piece_name))
    image, page_mungos, mdict = stack_images(images, mungos_per_page, mdict)

    # get only system mungos for unwrapping
    logging.info("{0}:\tSystem MuNGo loading".format(piece_name))
    system_mungos = [c for c in page_mungos if c.clsname == 'staff']
    system_mungos = sorted(system_mungos, key=lambda m: m.top)

    # unwrap sheet images
    logging.info("{0}:\tSheet image unrollilng".format(piece_name))
    un_wrapped_image, un_wrapped_coords = unwrap_sheet_image(image, system_mungos, mdict)

    # load performances
    logging.info("{0}:\tPerformance loading".format(piece_name))
    spectrograms = []
    midi_matrices = []
    onset_to_coord_maps = []

    for performance_key in piece.available_performances:

        # check if performance matches augmentation pattern
        tempo, synth = performance_key.split("tempo-")[1].split("_", 1)
        tempo = float(tempo) / 1000

        if synth not in aug_config["synths"]\
                or tempo < aug_config["tempo_range"][0]\
                or tempo > aug_config["tempo_range"][1]:
            continue

        # load current performance
        performance = piece.load_performance(performance_key, require_audio=require_audio)

        # running the alignment procedure
        alignment = align_score_to_performance(score, performance)

        # note events
        note_events = performance.load_note_events()

        # load spectrogram
        spec = performance.load_spectrogram()

        # compute onset to coordinate mapping
        onset_to_coord = onset_to_coordinates(alignment, un_wrapped_coords, note_events)
        onset_to_coord_maps.append(onset_to_coord)

        if load_midi_matrix:
            midi = performance.load_midi_matrix()

            if midi.shape[1] != spec.shape[1]:
                logging.debug('Perf {0}: Midi matrix and spectrogram have'
                              ' a different number of frames: MM {1}, spec {2}'
                              ''.format(performance_key, midi.shape[1], spec.shape[1]))
                n_frames = min(midi.shape[1], spec.shape[1])
                midi = midi[:, :n_frames]
                spec = spec[:, :n_frames]

            midi_matrices.append(midi)
        spectrograms.append(spec)

    if load_midi_matrix:
        return un_wrapped_image, spectrograms, onset_to_coord_maps, midi_matrices
    else:
        return un_wrapped_image, spectrograms, onset_to_coord_maps


def load_audio_score_retrieval():
    """
    Load alignment data
    """

    # Give piece dir
    collection_dir = '/home/matthias/cp/data/msmd'

    piece_names = ['BachCPE__cpe-bach-rondo__cpe-bach-rondo', 'BachJS__BWV259__bwv-259']

    all_piece_images = []
    all_piece_specs = []
    all_piece_o2c_maps = []
    for piece_name in piece_names:

        piece_image, piece_specs, piece_o2c_maps = prepare_piece_data(collection_dir, piece_name)

        # keep stuff
        all_piece_images.append(piece_image)
        all_piece_specs.append(piece_specs)
        all_piece_o2c_maps.append(piece_o2c_maps)

    return AudioScoreRetrievalPool(all_piece_images, all_piece_specs, all_piece_o2c_maps,
                                   data_augmentation=AUGMENT)


if __name__ == "__main__":
    """ main """

    pool = load_audio_score_retrieval()

    for i in range(100):
        sheet, spec = pool[i:i+1]

        plt.figure()
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(sheet[0, 0], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(spec[0, 0], cmap="viridis", origin="lower")
        plt.show()