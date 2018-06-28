
from __future__ import print_function

import logging
import os
import sys
import yaml
from tqdm import tqdm

from msmd.DEFAULT_CONFIG import DATA_ROOT_MSMD
from msmd.data_pools.data_pools import prepare_piece_data, prepare_piece_staff_data, StaffPool
from msmd.data_pools.data_pools import AudioScoreRetrievalPool, AUGMENT, NO_AUGMENT
from msmd.data_pools.data_pools import SPEC_CONTEXT, SHEET_CONTEXT_LEFT, SHEET_CONTEXT_RIGHT, SYSTEM_HEIGHT

from data_pools import ScoreInformedTranscriptionPool


def load_split(split_file):

    with open(split_file, 'rb') as hdl:
        split = yaml.load(hdl)

    return split


def load_piece_list(piece_names, aug_config=NO_AUGMENT):
    """
    Collect piece data
    """
    all_images = []
    all_specs = []
    all_o2c_maps = []
    for ip in tqdm(range(len(piece_names)), ncols=70):
        piece_name = piece_names[ip]

        try:
            image, specs, o2c_maps = prepare_piece_data(DATA_ROOT_MSMD, piece_name,
                                                        aug_config=aug_config, require_audio=False)
        except:
            print("Problems with loading piece %s" % piece_name)
            print(sys.exc_info()[0])
            continue

        # keep stuff
        all_images.append(image)
        all_specs.append(specs)
        all_o2c_maps.append(o2c_maps)

    return all_images, all_specs, all_o2c_maps


def load_piece_list_midi(piece_names, aug_config=NO_AUGMENT, data_root=None):
    """
    Collect piece data. Returns the images, spectrograms, onset-to-coord mappings,
    and midi matrices. For each piece in the given list (``piece_names``), there
    is one image, and as many spectrograms, onset-to-coord maps and midi matrices
    as there are performances for the piece. (This is because we currently only
    have one score per piece.)
    """
    if not data_root:
        data_root = DATA_ROOT_MSMD

    all_images = []
    all_specs = []
    all_o2c_maps = []
    all_midi_mats = []
    for ip in tqdm(range(len(piece_names)), ncols=70):
        piece_name = piece_names[ip]

        try:
            image, specs, o2c_maps, midi_mats = prepare_piece_data(data_root, piece_name, aug_config=aug_config,
                                                                   require_audio=False, load_midi_matrix=True)
        except Exception as e:
            print("Problems with loading piece %s" % piece_name)
            print(sys.exc_info()[0])
            raise e
            continue

        # keep stuff
        all_images.append(image)
        all_specs.append(specs)
        all_o2c_maps.append(o2c_maps)
        all_midi_mats.append(midi_mats)

    return all_images, all_specs, all_o2c_maps, all_midi_mats


def load_staff_list_midi(piece_names, aug_config, data_root):
    """
    Collect piece data. Returns the images, spectrograms, onset-to-coord mappings,
    and midi matrices. For each piece in the given list (``piece_names``), there
    is one image, and as many spectrograms, onset-to-coord maps and midi matrices
    as there are performances for the piece. (This is because we currently only
    have one score per piece.)

    Formally, this function behaves exactly the same as ``load_piece_list_midi()``.
    The important difference is that instead of returning the entity list for
    entire pieces, this splits the pieces into per-staff bundles of
    ``(image, specs, o2c_maps, midi_matrices)``. The data pool downstream
    will then consider each staff to be a piece. This means we don't
    """
    if not data_root:
        data_root = DATA_ROOT_MSMD

    all_images = []
    all_specs = []
    all_o2c_maps = []
    all_midi_mats = []
    for ip in tqdm(range(len(piece_names)), ncols=70):
        piece_name = piece_names[ip]

        try:
            per_staff_image, per_staff_specs, per_staff_o2c_maps, \
                per_staff_midi_mats = prepare_piece_staff_data(data_root, piece_name,
                                                               aug_config=aug_config,
                                                               require_audio=False,
                                                               load_midi_matrix=True)
        except Exception as e:
            print("Problems with loading piece %s" % piece_name)
            print(sys.exc_info()[0])
            raise e
            # continue

        print('Piece {}: {} performances, {} staffs'
              ''.format(piece_name, len(per_staff_image), len(per_staff_specs[0])))
        for _i_staff, staff_mms in enumerate(per_staff_midi_mats):
            print('\tS. {} MM shapes: {}'.format(_i_staff, [mm.shape for mm in staff_mms]))

        # keep stuff
        all_images.extend(per_staff_image)
        all_specs.extend(per_staff_specs)
        all_o2c_maps.extend(per_staff_o2c_maps)
        all_midi_mats.extend(per_staff_midi_mats)

    return all_images, all_specs, all_o2c_maps, all_midi_mats


##############################################################################
# These functions wrap the data pool initialization with appropriate config
# values.


def load_audio_score_retrieval(split_file, config_file=None, test_only=False):
    """
    Load alignment data
    """

    if not config_file:
        spec_context = SPEC_CONTEXT
        sheet_context = SHEET_CONTEXT_LEFT + SHEET_CONTEXT_RIGHT
        staff_height = SYSTEM_HEIGHT
        augment = AUGMENT
        no_augment = NO_AUGMENT
        test_augment = NO_AUGMENT.copy()
    else:
        with open(config_file, 'rb') as hdl:
            config = yaml.load(hdl)
        spec_context = config["SPEC_CONTEXT"]
        sheet_context = config["SHEET_CONTEXT"]
        staff_height = config["SYSTEM_HEIGHT"]
        augment = config["AUGMENT"]
        no_augment = NO_AUGMENT
        test_augment = NO_AUGMENT.copy()
        test_augment['synths'] = [config["TEST_SYNTH"]]
        test_augment['tempo_range'] = [config["TEST_TEMPO"], config["TEST_TEMPO"]]

    # selected pieces
    split = load_split(split_file)

    # initialize data pools
    if not test_only:
        tr_images, tr_specs, tr_o2c_maps = load_piece_list(split['train'], aug_config=augment)
        tr_pool = AudioScoreRetrievalPool(tr_images, tr_specs, tr_o2c_maps,
                                          spec_context=spec_context, sheet_context=sheet_context, staff_height=staff_height,
                                          data_augmentation=augment, shuffle=True)
        print("Train: %d" % tr_pool.shape[0])

        va_images, va_specs, va_o2c_maps = load_piece_list(split['valid'], aug_config=no_augment)
        va_pool = AudioScoreRetrievalPool(va_images, va_specs, va_o2c_maps,
                                          spec_context=spec_context, sheet_context=sheet_context, staff_height=staff_height,
                                          data_augmentation=no_augment, shuffle=False)
        va_pool.reset_batch_generator()
        print("Valid: %d" % va_pool.shape[0])

    else:
        tr_pool = va_pool = None

    te_images, te_specs, te_o2c_maps = load_piece_list(split['test'], aug_config=test_augment)
    te_pool = AudioScoreRetrievalPool(te_images, te_specs, te_o2c_maps,
                                      spec_context=spec_context, sheet_context=sheet_context, staff_height=staff_height,
                                      data_augmentation=no_augment, shuffle=False)
    print("Test: %d" % te_pool.shape[0])

    return dict(train=tr_pool, valid=va_pool, test=te_pool, train_tag="")


def load_score_informed_transcription(split_file, config_file=None, test_only=False,
                                      data_root=None, no_test=False):
    """
    Load alignment data
    """
    if not config_file:
        logging.warning('load_score_informed_transcription(): no config file provided, loading'
                        ' default values.')
        staff_height = SYSTEM_HEIGHT
        sheet_context_left = SHEET_CONTEXT_LEFT
        sheet_context_right = SHEET_CONTEXT_RIGHT
        sheet_context = sheet_context_left + sheet_context_right

        spec_context = SPEC_CONTEXT
        augment = AUGMENT
        no_augment = NO_AUGMENT
        va_augment = NO_AUGMENT.copy()
        test_augment = NO_AUGMENT.copy()
    else:
        with open(config_file, 'rb') as hdl:
            config = yaml.load(hdl)
        if "VA_AUGMENT" not in config:
            config["VA_AUGMENT"] = config["AUGMENT"]

        spec_context = config["SPEC_CONTEXT"]

        if "SHEET_CONTEXT_LEFT" in config:
            sheet_context_left = config["SHEET_CONTEXT_LEFT"]
            sheet_context_right = config["SHEET_CONTEXT_RIGHT"]
            sheet_context = sheet_context_left + sheet_context_right
        else:
            sheet_context = config["SHEET_CONTEXT"]
            sheet_context_left = sheet_context // 2
            sheet_context_right = sheet_context - sheet_context_left
        staff_height = config["SYSTEM_HEIGHT"]
        augment = config["AUGMENT"]
        no_augment = NO_AUGMENT
        va_augment = config["VA_AUGMENT"]
        test_augment = NO_AUGMENT.copy()
        test_augment['synths'] = [config["TEST_SYNTH"]]
        test_augment['tempo_range'] = [config["TEST_TEMPO"], config["TEST_TEMPO"]]

    # selected pieces
    split = load_split(split_file)

    # initialize data pools
    if not test_only:
        tr_images, tr_specs, tr_o2c_maps, tr_midis = load_piece_list_midi(split['train'],
                                                                          aug_config=augment,
                                                                          data_root=data_root)
        tr_pool = ScoreInformedTranscriptionPool(tr_images, tr_specs, tr_o2c_maps, tr_midis,
                                                 spec_context=spec_context,
                                                 sheet_context_left=sheet_context_left,
                                                 sheet_context_right=sheet_context_right,
                                                 staff_height=staff_height,
                                                 data_augmentation=augment, shuffle=True)
        print("Train: %d" % tr_pool.shape[0])

        va_images, va_specs, va_o2c_maps, va_midis = load_piece_list_midi(split['valid'],
                                                                          aug_config=va_augment,
                                                                          data_root=data_root)
        va_pool = ScoreInformedTranscriptionPool(va_images, va_specs, va_o2c_maps, va_midis,
                                                 spec_context=spec_context,
                                                 sheet_context_left=sheet_context_left,
                                                 sheet_context_right=sheet_context_right,
                                                 staff_height=staff_height,
                                                 data_augmentation=va_augment, shuffle=False)
        va_pool.reset_batch_generator()
        print("Valid: %d" % va_pool.shape[0])

    else:
        tr_pool = va_pool = None

    te_pool = None
    if not no_test:
        te_images, te_specs, te_o2c_maps, te_midis = load_piece_list_midi(split['test'], aug_config=test_augment, data_root=data_root)
        te_pool = ScoreInformedTranscriptionPool(te_images, te_specs, te_o2c_maps, te_midis,
                                                 spec_context=spec_context,
                                                 sheet_context_left=sheet_context_left,
                                                 sheet_context_right=sheet_context_right,
                                                 staff_height=staff_height,
                                                 data_augmentation=no_augment, shuffle=False)
        print("Test: %d" % te_pool.shape[0])

    return dict(train=tr_pool, valid=va_pool, test=te_pool, train_tag="")


def load_e2e_omr(split_file, config_file=None, test_only=False,
                                      data_root=None, no_test=False):
    """Loads the train, validate and test data pools for end-to-end OMR
    on staffs.

    :param split_file: A YAML file that defines which MSMD piees should go
        into the training, validation, and test set.

    :param config_file: The experiment configuration. Takes the relevant
        config values -- preprocessing, data augmentation.

    :param test_only: If set, only loads the test pool (the training and
        validation pools will have entries in the returned dict, with value
        ``None``).

    :param data_root: The MSMD root directory.

    :param no_test: If set, the testing pool will be ``None``.

    :return: ``dict(train=tr_pool, valid=va_pool, test=te_pool, train_tag="")``
    """

    if not config_file:
        logging.warning('load_score_informed_transcription(): no config file provided, loading'
                        ' default values.')
        staff_height = SYSTEM_HEIGHT
        sheet_context_left = SHEET_CONTEXT_LEFT
        sheet_context_right = SHEET_CONTEXT_RIGHT
        sheet_context = sheet_context_left + sheet_context_right

        spec_context = SPEC_CONTEXT
        augment = AUGMENT
        no_augment = NO_AUGMENT
        va_augment = NO_AUGMENT.copy()
        test_augment = NO_AUGMENT.copy()
    else:
        with open(config_file, 'rb') as hdl:
            config = yaml.load(hdl)
        if "VA_AUGMENT" not in config:
            config["VA_AUGMENT"] = config["AUGMENT"]

        spec_context = config["SPEC_CONTEXT"]

        if "SHEET_CONTEXT_LEFT" in config:
            sheet_context_left = config["SHEET_CONTEXT_LEFT"]
            sheet_context_right = config["SHEET_CONTEXT_RIGHT"]
            sheet_context = sheet_context_left + sheet_context_right
        else:
            sheet_context = config["SHEET_CONTEXT"]
            sheet_context_left = sheet_context // 2
            sheet_context_right = sheet_context - sheet_context_left
        staff_height = config["SYSTEM_HEIGHT"]
        augment = config["AUGMENT"]
        no_augment = NO_AUGMENT
        va_augment = config["VA_AUGMENT"]
        test_augment = NO_AUGMENT.copy()
        test_augment['synths'] = [config["TEST_SYNTH"]]
        test_augment['tempo_range'] = [config["TEST_TEMPO"], config["TEST_TEMPO"]]

    # selected pieces
    split = load_split(split_file)
    # initialize data pools
    if not test_only:
        tr_images, tr_specs, tr_o2c_maps, tr_midis = load_staff_list_midi(split['train'],
                                                                          aug_config=augment,
                                                                          data_root=data_root)
        tr_pool = StaffPool(tr_images,
                            tr_specs,
                            # tr_o2c_maps,
                            tr_midis,
                            # spec_context=spec_context,
                            # sheet_context_left=sheet_context_left,
                            # sheet_context_right=sheet_context_right,
                            staff_height=staff_height,
                            data_augmentation=augment, shuffle=True)
        print("Train: %d" % tr_pool.shape[0])

        va_images, va_specs, va_o2c_maps, va_midis = load_staff_list_midi(split['valid'],
                                                                          aug_config=va_augment,
                                                                          data_root=data_root)
        va_pool = StaffPool(va_images,
                            va_specs,
                            # va_o2c_maps,
                            va_midis,
                            # spec_context=spec_context,
                            # sheet_context_left=sheet_context_left,
                            # sheet_context_right=sheet_context_right,
                            staff_height=staff_height,
                            data_augmentation=va_augment, shuffle=False)
        va_pool.reset_batch_generator()
        print("Valid: %d" % va_pool.shape[0])

    else:
        tr_pool = va_pool = None

    te_pool = None
    if not no_test:
        te_images, te_specs, te_o2c_maps, te_midis = load_staff_list_midi(split['test'],
                                                                          aug_config=test_augment,
                                                                          data_root=data_root)
        te_pool = StaffPool(te_images,
                            te_specs,
                            # te_o2c_maps,
                            te_midis,
                            # spec_context=spec_context,
                            # sheet_context_left=sheet_context_left,
                            # sheet_context_right=sheet_context_right,
                            staff_height=staff_height,
                            data_augmentation=no_augment, shuffle=False)
        print("Test: %d" % te_pool.shape[0])

    return dict(train=tr_pool, valid=va_pool, test=te_pool, train_tag="")


##############################################################################


if __name__ == "__main__":
    """ main """
    # import matplotlib.pyplot as plt
    # from audio_sheet_retrieval.models.mutopia_ccal_cont_rsz import prepare
    #
    # def train_batch_iterator(batch_size=1):
    #     """ Compile batch iterator """
    #     from audio_sheet_retrieval.utils.batch_iterators import MultiviewPoolIteratorUnsupervised
    #     batch_iterator = MultiviewPoolIteratorUnsupervised(batch_size=batch_size, prepare=prepare, k_samples=None)
    #     return batch_iterator
    #
    # data = load_audio_score_retrieval(split_file="/home/matthias/cp/src/msmd/msmd/splits/bach_split.yaml",
    #                                   config_file="/home/matthias/cp/src/audio_sheet_retrieval/audio_sheet_retrieval/exp_configs/mutopia_no_aug.yaml",
    #                                   test_only=True)
    #
    # bi = train_batch_iterator(batch_size=5)
    #
    # iterator = bi(data["test"])
    #
    # # show some train samples
    # import time
    #
    # for epoch in xrange(1000):
    #     start = time.time()
    #     for i, (sheet, spec) in enumerate(iterator):
    #
    #         plt.figure()
    #         plt.clf()
    #
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(sheet[0, 0], cmap="gray")
    #         plt.ylabel(sheet[0, 0].shape[0])
    #         plt.xlabel(sheet[0, 0].shape[1])
    #         plt.colorbar()
    #
    #         plt.subplot(1, 2, 2)
    #         plt.imshow(spec[0, 0], cmap="viridis", origin="lower")
    #         plt.ylabel(spec[0, 0].shape[0])
    #         plt.xlabel(spec[0, 0].shape[1])
    #         plt.colorbar()
    #
    #         plt.show()

    import matplotlib.pyplot as plt


    def prepare(x, y, z):
        """ prepare images for training """
        import cv2
        import numpy as np

        # convert sheet snippet to float
        if isinstance(x, list):
            x = [x_i.astype(np.float32) / 255 for x_i in x]
        else:
            x = x.astype(np.float32)
            x /= 255

        # # resize sheet image
        # sheet_shape = [x.shape[2] // 2, x.shape[3] // 2]
        # new_shape = [x.shape[0], x.shape[1], ] + sheet_shape
        # x_new = np.zeros(new_shape, np.float32)
        # for i in range(len(x)):
        #    x_new[i, 0] = cv2.resize(x[i, 0], (sheet_shape[1], sheet_shape[0]))
        # x = x_new

        return x, y, z


    def train_batch_iterator(batch_size=1):
        """ Compile batch iterator """
        from msmd.data_pools.batch_iterators import TripleviewPoolIteratorUnsupervised
        from msmd.data_pools.batch_iterators import VariableLengthSequencePoolIterator
        batch_iterator = VariableLengthSequencePoolIterator(batch_size=batch_size, prepare=prepare, k_samples=None)
        return batch_iterator

    split_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "splits", "test_split.yaml")
    print('loading split: {}'.format(split_file))

    # Hack for experimental config
    config_file = "/home/matthias/cp/src/audio_sheet_retrieval/audio_sheet_retrieval/exp_configs/mutopia_no_aug.yaml"
    if not os.path.isfile(config_file):
        config_file = "/Users/hajicj/jku/gitlab/multimodal_transcription/multimodal_transcription/exp_configs/trans_mutopia_no_aug.yaml"
    if not os.path.isfile(config_file):
        raise OSError('Experiment config file mutopia_no_aug.yaml not found in any of the expected locations.')

    # data = load_score_informed_transcription(
    #     split_file=split_file,
    #     config_file=config_file,
    #     test_only=True)

    data = load_e2e_omr(
        split_file=split_file,
        config_file=config_file,
        test_only=True)

    bi = train_batch_iterator(batch_size=5)

    iterator = bi(data["test"])

    # show some train samples
    import time

    for epoch in xrange(1000):
        start = time.time()
        for i, (sheet, spec, midi) in enumerate(iterator):

            plt.figure()
            plt.clf()

            plt.subplot(3, 1, 1)
            plt.imshow(sheet[0][0], cmap="gray")
            plt.ylabel(sheet[0][0].shape[0])
            plt.xlabel(sheet[0][0].shape[1])
            plt.colorbar()

            plt.subplot(3, 1, 2)
            plt.imshow(spec[0][0], cmap="viridis", origin="lower", aspect="auto")
            plt.ylabel(spec[0][0].shape[0])
            plt.xlabel(spec[0][0].shape[1])
            plt.colorbar()

            plt.subplot(3, 1, 3)
            plt.imshow(midi[0][0], cmap="gray", origin="lower", interpolation="nearest", aspect="auto")
            plt.ylabel(midi[0][0].shape[0])
            plt.xlabel(midi[0][0].shape[1])
            plt.colorbar()

            plt.show()
