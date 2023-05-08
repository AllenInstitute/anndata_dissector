import argparse
import json
import pathlib


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_path',
        type=str,
        default=None,
        help='path to .csv with cell metadata')
    parser.add_argument(
        '--dst_path',
        type=str,
        default=None,
        help='path to file to be written')
    parser.add_argument(
        '--column_names',
        type=str,
        default=None,
        nargs='+',
        help='columns to keep in obs')
    parser.add_argument(
        '--subset_column',
        type=str,
        default=None,
        help='column used to denote different subsets')
    args = parser.parse_args()

    if isinstance(args.column_names, str):
        column_names = [args.column_names]
    else:
        column_names = args.column_names

    config = dict()

    with open(args.src_path, "r") as in_file:
        header = in_file.readline().strip().split(',')
        header_to_idx = {
            n:ii for ii, n in enumerate(header)}
        for line in in_file:
            params = line.strip().split(',')
            if len(params) != len(header):
                raise RuntimeError("line mismatch")
            feature = params[header_to_idx[args.subset_column]]
            if feature not in config:
                config[feature] = []
            current_config = config[feature]

            this_cell = {
                'row_idx': int(params[header_to_idx['source_row_index']])}
            for col in args.column_names:
                this_cell[col] = params[header_to_idx[col]]
            current_config.append(this_cell)

    with open(args.dst_path, 'w') as out_file:
        out_file.write(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
