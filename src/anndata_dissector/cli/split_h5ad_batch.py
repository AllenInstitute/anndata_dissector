import argparse
import json
import pathlib
import shutil
import tempfile

from anndata_dissector.utils.utils import (
    _clean_up)

from anndata_dissector.modules.split_h5ad import (
    extract_h5ad)


def split_h5ad_batch(
        parent_path,
        cell_config_lookup,
        gene_config,
        obs_index_column='cell_label',
        var_index_column='gene_identifier',
        output_dir=None,
        tmp_dir=None,
        clobber=False,
        layer_config=None):
    """
    layer_config is a list of dicts like
        {'layer': name of layer
         'norm': name of normalization
         'tag': tag appended to file name}
    """

    output_dir = pathlib.Path(output_dir)
    if not output_dir.is_dir():
        raise RuntimeError(
            f"{output_dir} is not dir")

    parent_path = pathlib.Path(parent_path)

    if not parent_path.is_file():
        raise RuntimeError(
            f"{parent_path} is not a file")

    original_parent = str(parent_path.resolve().absolute())

    if tmp_dir is not None:
        tmp_dir = pathlib.Path(tempfile.mkdtemp(dir=tmp_dir))
        new_path = tmp_dir / parent_path.name
        assert new_path != parent_path
        print(f"copying {parent_path} to {new_path}")
        shutil.copy(
            src=parent_path,
            dst=new_path)
        parent_path = new_path
        print("done copying")

    if layer_config is None:
        norm_lookup = {
            "rawcount": "raw",
            "X": "log2(CPM+1)"}
        tag_lookup = {
            "rawcount": "raw",
            "X": "log2"}
    else:
        norm_lookup = dict()
        tag_lookup = dict()
        for layer in layer_config:
            norm_lookup[layer['layer']] = layer['norm']
            tag_lookup[layer['layer']] = layer['tag']

    for config_key in cell_config_lookup:
        if len(config_key) == 0:
            continue
        cell_config = cell_config_lookup[config_key]
        for layer in norm_lookup:
            out_path = output_dir / f"{config_key}-{tag_lookup[layer]}.h5ad"
            metadata = {
                "parent": original_parent,
                "normalization": norm_lookup[layer]}
            extract_h5ad(
                parent_path=parent_path,
                cell_config=cell_config,
                gene_config=gene_config,
                layer=layer,
                output_path=out_path,
                obs_index_column=obs_index_column,
                var_index_column=var_index_column,
                metadata=metadata,
                clobber=clobber)
            print(f"wrote {out_path}")
    print("done writing files")

    if tmp_dir is not None:
        _clean_up(tmp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--parent_path',
        type=str,
        default=None,
        help='Path to the large h5ad file being split')
    parser.add_argument(
        '--cell_config_path',
        type=str,
        default=None,
        help='Path to the JSONized config showing how to split rows of '
        'h5ad file')
    parser.add_argument(
        '--gene_config_path',
        type=str,
        default=None,
        help='Path to JSONized config for gene data')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Path to directory where output files will be written')
    parser.add_argument(
        '--tmp_dir',
        type=str,
        default=None,
        help='Path to fast tmp_dir where parent file will be copied')
    parser.add_argument(
        '--clobber',
        default=False,
        action='store_true')
    parser.add_argument(
        '--is_merfish',
        default=False,
        action='store_true')

    args = parser.parse_args()

    cell_config_lookup = json.load(
        open(args.cell_config_path, 'rb'))
    gene_config = json.load(
        open(args.gene_config_path, 'rb'))

    layer_config = None
    if args.is_merfish:
        layer_config = [
            {'layer': 'X', 'norm': 'log2p', 'tag': 'log2'},
            {'layer': 'raw', 'norm': 'raw', 'tag': 'raw'}
        ]

    split_h5ad_batch(
        parent_path=args.parent_path,
        cell_config_lookup=cell_config_lookup,
        gene_config=gene_config,
        output_dir=args.output_dir,
        tmp_dir=args.tmp_dir,
        clobber=args.clobber,
        layer_config=layer_config)
