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
        help='columns to keep in var')
    args = parser.parse_args()

    config = []
    with open(args.src_path, "r") as in_file:
        header = in_file.readline().strip().split(',')
        header_to_idx = {
           n:ii for ii, n in enumerate(header)}
        for line in in_file:
            params = line.strip().split(',')
            this = {
                col: params[header_to_idx[col]]
                for col in args.column_names
            }
            config.append(this)
    with open(args.dst_path, 'w') as out_file:
        out_file.write(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
