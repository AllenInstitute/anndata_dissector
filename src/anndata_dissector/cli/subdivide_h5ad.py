import argparse

from anndata_dissector.modules.split_h5ad import (
    subdivide_h5ad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_path',
        type=str,
        default=None,
        help='The h5ad file to be subdivided')
    parser.add_argument(
        '--n_sub',
        type=int,
        default=None,
        help='The number of subdivisions')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='The output directory')
    parser.add_argument(
        '--clobber',
        default=False,
        action='store_true',
        help='Overwrite existing files where relevant')

    args = parser.parse_args()

    subdivide_h5ad(
        parent_path=args.src_path,
        number_of_subdivisions=args.n_sub,
        output_dir=args.output_dir,
        clobber=args.clobber)


if __name__ == "__main__":
    main()
