import anndata
import copy
import gc
import h5py
import pandas as pd
import pathlib
import scipy.sparse as scipy_sparse


from anndata_dissector.utils.pandas_utils import (
    read_df_from_h5ad)

from anndata_dissector.utils.sparse_utils import (
    load_disjoint_csr)


def extract_h5ad(
        parent_path,
        cell_config,
        gene_config,
        layer,
        output_path,
        obs_index_column,
        var_index_column,
        metadata=None,
        clobber=False,
        tmp_dir=None):
    """
    Write out a subset of rows from a parent h5ad file
    to a smaller h5ad file.

    Parameters
    ----------
    parent_path:
        Path to the large h5ad file
    cell_config:
        List of dicts, each represnting a cell to load
        {'row_idx': row in parent file
         ....key value pairs for obs_df...}
    gene_config:
        List of dicts representing data for var df
    layer:
        if 'X', extract from parent.X; else extract from
        parent.layers.layer
    output_path:
        file to write
    obs_index_column:
        column in obs df that will be the index
    var_index_column:
        column in var df that will be the index
    metadata:
       optional unstructured metadata
    clobber:
        if False, not overwrite existing output_path
    """
    parent_path = pathlib.Path(parent_path)
    if not parent_path.is_file():
        raise RuntimeError(
            f"{parent_path} is not a file")

    output_path = pathlib.Path(output_path)
    if not clobber:
        if output_path.exists():
            raise RuntimeError(
                f"{output_path} exists; "
                "run with clobber=True to overwrite")

    print(f"preparing to write {output_path}")
    print(f"{len(cell_config)} rows")

    original_var = read_df_from_h5ad(
        h5ad_path=parent_path,
        df_name='var')

    if len(original_var) != len(gene_config):
        raise RuntimeError(
            f"parent has {len(original_var)} genes\n"
            f"you specified {len(gene_config)}\n"
            "these must be equal")

    row_list = [cell['row_idx'] for cell in cell_config]

    if layer == 'X':
        data_key = 'X'
    else:
        data_key = f'layers/{layer}'

    if metadata is None:
        metadata = dict()
    else:
        metadata = copy.deepcopy(metadata)

    if 'parent' not in metadata:
        metadata['parent'] = str(parent_path.resolve().absolute())
    metadata['parent_layer'] = data_key
    metadata['parent_rows'] = copy.deepcopy(row_list)

    with h5py.File(parent_path, 'r', swmr=True) as src:
        attr_dict = dict(src[f'{data_key}'].attrs)
        is_csr = True
        if 'ecnoding-type' not in attr_dict:
            is_csr = False
        elif 'csr' not in attr_dict['encoding-type']:
            is_csr = False
        if not is_csr:
            raise RuntimeError(
                f"Not obvious that {parent_path}/{data_key} is CSR;\n"
                f"X attrs:\n{attr_dict}")

        (data,
         indices,
         indptr) = load_disjoint_csr(
                         row_index_list=row_list,
                         data=src[f'{data_key}/data'],
                         indices=src[f'{data_key}/indices'],
                         indptr=src[f'{data_key}/indptr'],
                         tmp_dir=tmp_dir)

    x_matrix = scipy_sparse.csr_matrix(
            (data, indices, indptr),
            shape=(len(cell_config), len(gene_config)))

    obs_data = []
    for cell in cell_config:
        new_cell = copy.deepcopy(cell)
        new_cell.pop('row_idx')
        obs_data.append(new_cell)
    obs_df = pd.DataFrame(obs_data)
    obs_df = obs_df.set_index(obs_index_column)

    var_df = pd.DataFrame(gene_config)
    var_df = var_df.set_index(var_index_column)

    a_data = anndata.AnnData(
        X=x_matrix,
        obs=obs_df,
        var=var_df,
        uns=metadata)

    a_data.write_h5ad(output_path)
    gc.collect()
