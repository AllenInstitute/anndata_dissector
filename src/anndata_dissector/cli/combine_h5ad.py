"""
Module to combine h5ad files row-wise

Config will just have
{'h5ad_list': [list of paths to h5ad files being combined],
 'output_path': path/to/output.h5ad}
"""
import argparse
import json

from anndata_dissector.modules.combine_h5ad import (
    combine_h5ad_row_wise)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--clobber', default=False, action='store_true')
    parser.add_argument('--tmp_dir', type=str, default=None)
    args = parser.parse_args()

    config = json.load(open(args.config_path, 'rb'))
    combine_h5ad_row_wise(
        h5ad_path_list=config['h5ad_path_list'],
        output_path=config['output_path'],
        clobber=args.clobber,
        tmp_dir=args.tmp_dir)


if __name__ == "__main__":
    main()
