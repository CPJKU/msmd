
import yaml
import numpy as np
from data_pools import prepare_piece_data
from prepare_random_split import collect_set


compute_note_counts = False


if __name__ == "__main__":
    """ main """

    collection_dir = '/media/matthias/Data/msmd/'
    piece_file = "/home/matthias/cp/src/msmd/msmd/splits/all_pieces.yaml"
    count_file = piece_file.replace(".yaml", "_counts.yaml")

    split_file = "/home/matthias/cp/src/msmd/msmd/splits/bach_out_split.yaml"

    # compute note counts for piece list
    if compute_note_counts:

        with open(piece_file, 'rb') as hdl:
            pieces = yaml.load(hdl)

        note_counts = dict()
        for i, piece_name in enumerate(pieces["success"]):
            print "%d / %d" % (i+1, len(pieces["success"]))
            _, _, piece_o2c_maps = prepare_piece_data(collection_dir, piece_name)
            count = len(piece_o2c_maps[0])
            note_counts[piece_name] = count

        with open(count_file, 'wb') as hdl:
            yaml.dump(note_counts, hdl)

    # load note counts
    else:
        with open(count_file, 'rb') as hdl:
            note_counts = yaml.load(hdl)

    # collect split
    # -------------

    composer = "BachJS"

    n_pieces = len(note_counts)
    n_onsets = np.sum(note_counts.values())

    print "%d pieces with %d onsets available" % (n_pieces, n_onsets)

    piece_names = np.asarray(note_counts.keys())
    counts = np.asarray(note_counts.values())

    sorted_idx = np.argsort(counts)
    piece_names = piece_names[sorted_idx]
    counts = counts[sorted_idx]

    # mark available pieces
    available = np.ones_like(counts, dtype=np.bool)

    # validation set is the rest
    pieces_te = []
    collected_te = 0
    for i_piece, piece in enumerate(piece_names):
        if composer in piece:
            pieces_te.append(piece)
            collected_te += note_counts[piece]
            available[i_piece] = False

    # collect train and test pieces
    # (contains 90% of remaining onsets)
    n_tr = 0.98 * (n_onsets - collected_te)
    pieces_tr, collected_tr, available = collect_set(piece_names, counts, available, n_tr, favour_long=True)

    # validation set is the rest
    pieces_va = []
    collected_va = 0
    for piece in piece_names:
        if piece not in pieces_tr + pieces_te:
            pieces_va.append(piece)
            collected_va += note_counts[piece]

    print "tr: %d pieces with %d onsets collected" % (len(pieces_tr), collected_tr)
    print "te: %d pieces with %d onsets collected" % (len(pieces_te), collected_te)
    print "va: %d pieces with %d onsets collected" % (len(pieces_va), collected_va)

    # dump split
    split = dict()
    split["train"] = [str(p) for p in pieces_tr]
    split["valid"] = [str(p) for p in pieces_va]
    split["test"] = [str(p) for p in pieces_te]

    with open(split_file, 'wb') as hdl:
        yaml.dump(split, hdl, default_flow_style=False)
