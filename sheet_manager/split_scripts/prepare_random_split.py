
import yaml
import numpy as np
# TODO: get this back from audio sheet retrieval
from audio_sheet_retrieval.utils.mutopia_data import prepare_piece_data


compute_note_counts = False

# bach
# ratio_te = 0.2
# ratio_va = 0.1

# all
ratio_te = 0.08
ratio_va = 0.02

ratio_tr = 1.0 - (ratio_te + ratio_va)


def strip_str_overlap(string_list, start_idx=0):
    stripped_list = list(string_list)
    min_len = np.min([len(s) for s in string_list])

    to_strip = ""
    for i_char in range(start_idx, min_len):
        all_equal = True
        char = string_list[0][i_char]
        for i_string in range(len(string_list)):
            if string_list[i_string][i_char] != char:
                all_equal = False
                break
        if all_equal:
            to_strip += char

    # print "start_idx:", start_idx
    # print "to_strip: ", to_strip

    if to_strip == "" and start_idx < min_len:
        stripped_list = strip_str_overlap(stripped_list, start_idx+1)
    elif to_strip != "":
        stripped_list = [s.strip(to_strip) for s in string_list]
        stripped_list = strip_str_overlap(stripped_list, 0)
    else:
        pass

    return stripped_list


def string_overlap(s1, s2):
    import difflib
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2))
    # overlap = s1[pos_a:pos_a + size]
    return size


def collect_set(piece_names, counts, available, n_target, favour_long=False):
    """ collect pieces """

    pieces = []
    collected = 0

    while collected < n_target:

        # get candidate pieces
        candidates = np.nonzero(available)[0]

        # randomly select pieces
        if favour_long:
            probs = counts[candidates].astype(np.float) / counts[candidates].sum()
            selected = np.random.choice(candidates, p=probs)
        else:
            selected = np.random.choice(candidates)

        # # compute maximum overlap
        # max_overlap = 0.0
        # min_overlap = np.inf
        # for cand_piece in piece_names[candidates]:
        #     for train_piece in pieces:
        #         max_overlap = max(max_overlap, string_overlap(cand_piece, train_piece))
        #         min_overlap = min(min_overlap, string_overlap(cand_piece, train_piece))
        #
        # sel_overlap = 0.0
        # for train_piece in pieces:
        #     sel_overlap = max(sel_overlap, string_overlap(piece_names[selected], train_piece))
        # print min_overlap, max_overlap, sel_overlap
        # print piece_names[selected]

        # select piece
        available[selected] = False

        pieces.append(piece_names[selected])
        collected += counts[selected]

    return pieces, collected, available


if __name__ == "__main__":
    """ main """

    collection_dir = '/media/matthias/Data/msmd/'
    piece_file = "/home/matthias/cp/src/sheet_manager/sheet_manager/splits/all_pieces.yaml"
    count_file = piece_file.replace(".yaml", "_counts.yaml")

    split_file = "/home/matthias/cp/src/sheet_manager/sheet_manager/splits/all_split.yaml"

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

    # find tr / va / te split
    # -----------------------

    n_pieces = len(note_counts)
    n_onsets = np.sum(note_counts.values())

    print "%d pieces with %d onsets available" % (n_pieces, n_onsets)

    n_te = ratio_te * n_onsets
    n_va = ratio_va * n_onsets
    n_tr = ratio_tr * n_onsets

    piece_names = np.asarray(note_counts.keys())
    counts = np.asarray(note_counts.values())

    sorted_idx = np.argsort(counts)
    piece_names = piece_names[sorted_idx]
    counts = counts[sorted_idx]

    # for i in range(n_pieces):
    #     print counts[i], piece_names[i]

    # mark available pieces
    available = np.ones_like(counts, dtype=np.bool)

    # collect train and test pieces
    pieces_tr, collected_tr, available = collect_set(piece_names, counts, available, n_tr, favour_long=True)
    pieces_te, collected_te, available = collect_set(piece_names, counts, available, n_te, favour_long=False)

    # validation set is the rest
    pieces_va = []
    collected_va = 0
    for piece in piece_names:
        if piece not in pieces_tr + pieces_te:
            pieces_va.append(piece)
            collected_va += note_counts[piece]

    # compute maximum overlap of train and test set
    striped_pieces_te = strip_str_overlap(pieces_te)
    striped_pieces_tr = strip_str_overlap(pieces_tr)

    max_overlap = 0.0
    min_overlap = np.inf
    for i_te, te_piece in enumerate(striped_pieces_te):
        for i_tr, tr_piece in enumerate(striped_pieces_tr):
            overlap = string_overlap(te_piece, tr_piece)

            if overlap > 10:
                print pieces_te[i_te]
                print pieces_tr[i_tr]
                print "-" * 50

            max_overlap = max(max_overlap, overlap)
            min_overlap = min(min_overlap, overlap)

    print "tr: %d pieces with %d (%d) onsets collected" % (len(pieces_tr), collected_tr, n_tr)
    print "te: %d pieces with %d (%d) onsets collected" % (len(pieces_te), collected_te, n_te)
    print "va: %d pieces with %d (%d) onsets collected" % (len(pieces_va), collected_va, n_va)

    # dump split
    split = dict()
    split["train"] = [str(p) for p in pieces_tr]
    split["valid"] = [str(p) for p in pieces_va]
    split["test"] = [str(p) for p in pieces_te]

    with open(split_file, 'wb') as hdl:
        yaml.dump(split, hdl, default_flow_style=False)
