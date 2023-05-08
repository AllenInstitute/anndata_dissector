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
        clobber=False):

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

    norm_lookup = {
        "rawcount": "raw",
        "X": "log2(CPM+1)"}
    tag_lookup = {
        "rawcount": "raw",
        "X": "log2"}

    for config_key in cell_config_lookup:
        cell_config = cell_config_lookup[config_key]
        for layer in ("X", "rawcount"):
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
