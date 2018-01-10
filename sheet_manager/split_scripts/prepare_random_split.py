
from __future__ import print_function

import yaml
import numpy as np


if __name__ == "__main__":
    """ main """

    piece_file = "/home/matthias/cp/src/sheet_manager/sheet_manager/splits/bach_pieces.yaml"
    count_file = piece_file.replace(".yaml", "_counts.yaml")

    split_file = "/home/matthias/cp/src/sheet_manager/sheet_manager/splits/bach_split.yaml"

    # load note counts
    with open(count_file, 'rb') as hdl:
        note_counts = yaml.load(hdl)

    # load existing split
    with open(split_file, 'rb') as hdl:
        split = yaml.load(hdl)

    # train note counts
    n_train = 0
    for p in split["train"]:
        n_train += note_counts[p]
    print("Original train notes:", n_train)
    print("Original piece count:", len(split["train"]))

    # ratio of notes to keep
    keep_ratio = 0.10

    # target onsets
    n_target = n_train * keep_ratio

    # new train data
    train_set = list(split["train"])
    n_reduced = n_train

    # remove pieces until target note count is reached
    while (0.95 * n_reduced) > n_target:
        piece = np.random.choice(train_set)
        train_set.remove(piece)
        n_reduced -= note_counts[piece]

    print("New train notes:", n_reduced)
    print("New piece count:", len(train_set))

    # new split
    new_split = split
    new_split["train"] = train_set

    # write dot yaml
    file_name = "bach_split_%02d.yaml" % (100 * keep_ratio)
    new_split_file = split_file.replace("bach_split.yaml", file_name)
    with open(new_split_file, 'wb') as hdl:
        yaml.dump(new_split, hdl, default_flow_style=False)
