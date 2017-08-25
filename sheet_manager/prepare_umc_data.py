#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 10:13:20 2017

@author: matthias
"""

from __future__ import print_function

import re
import os
import cv2
import glob
import yaml
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from sklearn.metrics.pairwise import pairwise_distances

from collections import defaultdict

from midi_parser import MidiParser, notes_to_onsets, FPS
from score_alignment.lilypond_note_coords.render_audio import render_audio

from lasagne_wrapper.utils import BColors
col = BColors()

# ROOT_DIR = "/media/matthias/Data/Data/umc_export/"
ROOT_DIR = "/media/matthias/Data/umc_export/"

# DST_DIR = "/home/matthias/cp/data/sheet_localization/umc/"
DST_DIR = "/home/matthias/mounts/home@rechenknecht0/Data/sheet_localization/umc/"

SHOW_PLOTS = False

LOG_FILE = "umc_data_prep.log"


if __name__ == '__main__':

    """ main """

    # open log file
    log_file = open(LOG_FILE, 'wb')

    # init
    note_count = 0
    note_count_annotated = 0
    piece_count = 0

    # get list of all pieces
    piece_dirs = np.sort(glob.glob(os.path.join(ROOT_DIR, '*')))
    n_pieces = len(piece_dirs)

    # iterate pieces
    for i_piece, piece_dir in enumerate(piece_dirs):
        piece_name = piece_dir.split('/')[-1]
        if "256_" not in piece_name:
            continue
        print(col.print_colored("\nProcessing piece %d of %d (%s)" % (i_piece + 1, n_pieces, piece_name), col.OKBLUE))

        # write log message
        log_file.write("\n%s\n" % piece_name)

        # midi file path
        midi_file_path = os.path.join(piece_dir, "score_ppq.mid")

        # load notes
        with open(os.path.join(piece_dir, "score_notes.yaml"), 'rb') as fp:
            yaml_notes = yaml.load(fp)

        # load systems
        page_systems = defaultdict(list)
        page_system_quaters = defaultdict(list)
        system_path = os.path.join(piece_dir, "score_systems.yaml")
        if os.path.exists(system_path):
            print("Systems annotated!")
            with open(system_path, 'rb') as fp:
                yaml_systems = yaml.load(fp)

                for yaml_system in yaml_systems:
                    page_id = yaml_system['page']

                    # convert system coordinates to array
                    system_bbox = np.zeros((4, 2))
                    system_bbox[0] = np.asarray([yaml_system['topLeft']])
                    system_bbox[1] = np.asarray([yaml_system['topRight']])
                    system_bbox[2] = np.asarray([yaml_system['bottomRight']])
                    system_bbox[3] = np.asarray([yaml_system['bottomLeft']])
                    system_bbox = system_bbox[:, ::-1]

                    # keep coordinate if system is not there
                    system_found = False
                    for i, bbox in enumerate(page_systems[page_id]):

                        # compute overlap in y-direction
                        dy_min = min(bbox[2, 0], system_bbox[2, 0]) - max(bbox[1, 0], system_bbox[1, 0])
                        dy_max = max(bbox[2, 0], system_bbox[2, 0]) - min(bbox[1, 0], system_bbox[1, 0])
                        if dy_min >= 0:
                            overlap = dy_min / dy_max
                        else:
                            overlap = 0

                        if overlap == 1:
                            system_found = True
                        elif overlap > 0:
                            system_found = True

                            # merge system coordinates
                            system_bbox[0, 1] = min(bbox[0, 1], system_bbox[0, 1])
                            system_bbox[1, 1] = max(bbox[1, 1], system_bbox[1, 1])
                            system_bbox[2, 1] = max(bbox[1, 1], system_bbox[1, 1])
                            system_bbox[3, 1] = min(bbox[0, 1], system_bbox[0, 1])
                            page_systems[page_id][i] = system_bbox

                            # append quarters covered by system
                            page_system_quaters[page_id][i].append(yaml_system['quarters'])

                    if not system_found:
                        page_systems[page_id].append(system_bbox)
                        page_system_quaters[page_id].append([yaml_system['quarters']])

                # convert coordinates to array
                for page_id in page_systems.keys():
                    page_systems[page_id] = np.asarray(page_systems[page_id], dtype=np.float32)

        else:
            print("No systems annotated!")
            continue

        # iterate notes
        note_pos_found = False
        notes_assigned = True
        page_note_coords = defaultdict(list)
        page_note_system_assignment = defaultdict(list)
        onsets = []
        for note in yaml_notes:
            if "pos" in note:
                note_pos_found = True

                # keep coordinate and note onsets
                duration = note["quarters"][1] - note["quarters"][0]
                if note['pos'][::-1] not in page_note_coords[note['page']]\
                        and duration > 0:

                    # assign note to system
                    found = 0
                    onset_quarter = note['quarters'][0]
                    page_system_quarters = page_system_quaters[note['page']]
                    for system_id, system_quarters in enumerate(page_system_quarters):
                        for quarters in system_quarters:
                            if quarters[0] <= onset_quarter < quarters[1]:
                                page_note_system_assignment[note['page']].append(system_id)
                                found += 1

                    # keep note and onset
                    if found == 1:
                        note_count += 1
                        page_note_coords[note['page']].append(note['pos'][::-1])
                        onsets.append(note['seconds'][0])
                    else:
                        notes_assigned = False
                        txt = "#could not assign note to system!"
                        print(col.print_colored(txt, col.FAIL))
                        log_file.write(txt + "\n")

        if not notes_assigned:
            continue

        if note_pos_found:
            piece_count += 1
            print("Notes found!")

            # convert coordinates to array
            for page_id in page_note_coords.keys():
                page_note_coords[page_id] = np.asarray(page_note_coords[page_id], dtype=np.float32)

        else:
            print("No notes annotated!")
            continue

        # load pages
        pages = []
        page_paths = np.sort(glob.glob(os.path.join(piece_dir, "pages/*.png")))
        for i_page, page_path in enumerate(page_paths):
            page_id = i_page + 1
            I = cv2.imread(page_path, 0)

            # resize image
            width = 835
            scale = float(width) / I.shape[1]
            height = int(scale * I.shape[0])
            I = cv2.resize(I, (width, height))
            pages.append(I)

            # re-scale coordinates
            page_note_coords[page_id] *= scale
            page_systems[page_id] *= scale

            # show sheet image and annotations
            if SHOW_PLOTS:
                plt.figure("sheet")
                plt.clf()
                ax = plt.subplot(111)

                # plot sheet
                plt.imshow(I, cmap=plt.cm.gray)
                plt.xlim([0, I.shape[1] - 1])
                plt.ylim([I.shape[0] - 1, 0])

                # plot system centers
                system_centers = np.mean(page_systems[page_id][:, [0, 3], 0], axis=1, keepdims=True)
                plt.plot(page_systems[page_id][:, 0, 1], system_centers.flatten(), 'mo')

                # plot notes
                plt.plot(page_note_coords[page_id][:, 1], page_note_coords[page_id][:, 0], 'co')

                # plot systems
                patches = []
                for system_coords in page_systems[page_id]:
                    polygon = Polygon(system_coords[:, ::-1], True)
                    patches.append(polygon)
                p = PatchCollection(patches, color='r', alpha=0.2)
                ax.add_collection(p)

                plt.show(block=True)

        # shift note coordinates towards system center
        for page_id in page_note_coords.keys():

            # distance to system centers (y-direction)
            system_centers = np.mean(page_systems[page_id][:, [0, 3], 0], axis=1, keepdims=True)
            system_heights = page_systems[page_id][:, 2, 0] - page_systems[page_id][:, 1, 0]
            dists = np.abs(pairwise_distances(page_note_coords[page_id][:, 0:1], system_centers))

            # shift notes to far away from system
            for i_note in xrange(page_note_coords[page_id].shape[0]):

                # assigned system id of note
                system_id = page_note_system_assignment[page_id][i_note]

                # check for maximum distance allowed
                dist_thresh = 0.50 * (system_heights[system_id] / 2 + 30)
                if dists[i_note, system_id] >= dist_thresh:
                    page_note_coords[page_id][i_note, 0] = system_centers[system_id]
                    txt = "note shifted on page %d towards system %d" % (page_id, system_id + 1)
                    print(col.print_colored(txt, col.WARNING))

        # count annotated notes
        note_count_annotated += np.sum([len(v) for v in page_note_coords.itervalues()])

        # dump sheet_manager compatible folder
        piece_folder_name = piece_name.replace(".", "").replace(",", "").replace(" ", "_").replace("-", "_")
        piece_folder_name = re.sub("_+", "_", piece_folder_name)
        piece_folder = os.path.join(DST_DIR, piece_folder_name)
        if not os.path.exists(piece_folder):
            os.makedirs(piece_folder)

        # copy midi file
        new_midi_file_path = os.path.join(piece_folder, "%s.mid" % piece_folder_name)
        shutil.copy(midi_file_path, new_midi_file_path)

        # save images and coordinates
        if not os.path.exists(os.path.join(piece_folder, "sheet")):
            os.makedirs(os.path.join(piece_folder, "sheet"))

        if not os.path.exists(os.path.join(piece_folder, "coords")):
            os.makedirs(os.path.join(piece_folder, "coords"))

        if not os.path.exists(os.path.join(piece_folder, "audio")):
            os.makedirs(os.path.join(piece_folder, "audio"))

        if not os.path.exists(os.path.join(piece_folder, "spec")):
            os.makedirs(os.path.join(piece_folder, "spec"))

        for i_page, page in enumerate(pages):
            page_id = i_page + 1

            img_file = os.path.join(piece_folder, "sheet", "%02d.png" % page_id)
            cv2.imwrite(img_file, page)

            coord_file = os.path.join(piece_folder, "coords", "notes_%02d.npy" % page_id)
            np.save(coord_file, page_note_coords[page_id])

            coord_file = os.path.join(piece_folder, "coords", "systems_%02d.npy" % page_id)
            np.save(coord_file, page_systems[page_id])

        # render audio
        audio_midi_file_path = os.path.join(piece_folder, "audio", "%s_bpm_100_Acoustic_Piano.mid" % piece_folder_name)
        shutil.copy(new_midi_file_path, audio_midi_file_path)
        render_audio(new_midi_file_path, sound_font="Acoustic_Piano", bpm_factor=100,
                     bpm=None, velocity=None, change_tempo=False)

        # compute spectrogram
        audio_file_path = audio_midi_file_path.replace(".mid", ".flac")
        midi_parser = MidiParser(show=False)
        spec, _ = midi_parser.process(midi_file_path, audio_file_path)

        # save onsets
        onsets = np.asarray(onsets)[:, np.newaxis]
        onsets = notes_to_onsets(onsets, dt=1.0 / FPS)
        file_name = os.path.basename(audio_file_path).split('.')[0]
        onset_file_path = os.path.join(piece_folder, "spec", file_name + '_onsets.npy')
        np.save(onset_file_path, onsets)

        # save spectrogram
        spec_file_path = os.path.join(piece_folder, "spec", file_name + '_spec.npy')
        np.save(spec_file_path, spec)

        log_file.write("#annotations exported for training!\n")
        log_file.flush()

    print("")
    print("%d aligned notes found in %d of %d pieces." % (note_count, piece_count, n_pieces))
    print("where %d can be used for deep learning." % note_count_annotated)

    # close log file
    log_file.close()
