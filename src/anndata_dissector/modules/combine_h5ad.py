import anndata
import h5py
import pandas as pd
import scipy.sparse as scipy_sparse
import tempfile

from anndata_dissector.utils.utils import (
    mkstemp_clean,
    _clean_up)

from anndata_dissector.utils.pandas_utils import (
    read_df_from_h5ad)

from anndata_dissector.utils.sparse_utils import (
    merge_csr_from_disk)


def combine_h5ad_row_wise(
        h5ad_path_list,
        output_path,
        tmp_dir=None):
    """
    Combine a list of h5ad files (assuming they are stored as CSR
    matrices)

    Parameters
    ----------
    h5ad_path_list:
        Ordered list of the h5ad files to combine
    output_path:
        Path to the new h5ad file
    tmp_dir:
        Directory where temporary hdf5 file containing the new sparse
        matrix will be stored.
    """

    tmp_dir = tempfile.mkdtemp(
        dir=tmp_dir,
        prefix='h5ad_combination_')
    h5_tmp_path = mkstemp_clean(
        dir=tmp_dir,
        suffix='.h5')

    var = None
    obs_data = []
    obs_idx_name = None

    uns = None
    uns_assigned = False

    for pth in h5ad_path_list:
        this_var = read_df_from_h5ad(pth, 'var')
        this_obs = read_df_from_h5ad(pth, 'obs')
        this_uns = read_df_from_h5ad(pth, 'uns')

        if not uns_assigned:
            uns = this_uns
            uns_assigned = True
        elif uns is not None:
            k_list = list(uns.keys())
            for k in this_uns:
                if k not in uns:
                    continue
                if this_uns[k] != uns[k]:
                    uns.pop(k)
            for k in k_list:
                if k not in this_uns:
                    uns.pop(k)

        if obs_idx_name is None:
            obs_idx_name = this_obs.index.name
        this_obs = this_obs.reset_index()
        if var is None:
            var = this_var
            baseline_var_pth = pth
        else:
            if not var.equals(this_var):
                raise RuntimeError(
                    f"{pth}\nhas different var dataframe than\n"
                    f"{baseline_var_pth}")
        obs_data += this_obs.to_dict(orient='records')

    merge_csr_from_disk(
        h5ad_path_list=h5ad_path_list,
        output_path=h5_tmp_path)

    with h5py.File(h5_tmp_path, 'r') as src:
        csr_matrix = scipy_sparse.csr_matrix(
             (src['data'][()],
              src['indices'][()],
              src['indptr'][()]),
             shape=(len(obs_data), len(var)))

    a_data = anndata.AnnData(
        X=csr_matrix,
        var=var,
        obs=pd.DataFrame(obs_data).set_index(obs_idx_name),
        uns=uns)

    a_data.write_h5ad(output_path)

    _clean_up(tmp_dir)
