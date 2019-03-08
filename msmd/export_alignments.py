"""This script takes a path to a dataset as input
and exports the alignments from a mung-file to csv files.
"""
import os
import msmd
import argparse
import pandas as pd


def main(path_data, key_ref, key_des):
    for cur_piece_name in os.listdir(path_data):
        if os.path.isdir(os.path.join(path_data, cur_piece_name)):
            # piece loading
            piece = msmd.data_model.piece.Piece(root=path_data, name=cur_piece_name)
            performance_names = piece.available_performances

            ref_idx = [idx for idx, cur_name in enumerate(performance_names)
                       if key_ref in cur_name]
            des_idx = [idx for idx, cur_name in enumerate(performance_names)
                       if key_des in cur_name]

            if len(ref_idx) == 0 or len(des_idx) == 0:
                continue

            ref_idx = ref_idx[0]
            des_idx = des_idx[0]

            aln = piece.load_pairwise_alignment(performance_ref=performance_names[ref_idx],
                                                performance_des=performance_names[des_idx])
            aln = pd.DataFrame(aln, columns=[performance_names[ref_idx], performance_names[des_idx]])
            cur_fn = 'aln_{}-{}.csv'.format(performance_names[ref_idx], performance_names[des_idx])
            aln.to_csv(cur_fn, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export alignments.')
    parser.add_argument('--data', help='Path to the dataset.')
    parser.add_argument('--key_ref', help='Performance key of the reference.')
    parser.add_argument('--key_des', help='Performance key of the desired performance.')
    args = parser.parse_args()

    main(args.data, args.key_ref, args.key_des)
