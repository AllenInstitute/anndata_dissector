import anndata
import numpy as np
import pandas as pd

from anndata_dissector.utils.utils import (
    mkstemp_clean)

from anndata_dissector.utils.pandas_utils import (
    read_df_from_h5ad)


def test_read_df_from_h5ad(tmp_dir_fixture):
    """
    Test that we can read the obs and var dataframes
    from h5ad by directly calling anndata's backend
    """
    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='read_df_data_',
        suffix='.h5ad')

    n_rows = 223
    n_cols = 114
    rng = np.random.default_rng(77123)
    X = rng.random((n_rows, n_cols))
    obs_data = [
        {'cell_id': f'cell_{ii}',
         'random': rng.integers(0, 99),
         'not_random': 7*ii}
        for ii in range(n_rows)]
    obs_df = pd.DataFrame(obs_data)
    obs_df = obs_df.set_index('cell_id')

    var_data = [
        {'gene_name': f'g_{ii}',
         'random': rng.integers(111, 222),
         'not_random': ii**2}
        for ii in range(n_cols)]

    var_df = pd.DataFrame(var_data)
    var_df = var_df.set_index('gene_name')

    a_data = anndata.AnnData(X=X, obs=obs_df, var=var_df)
    a_data.write_h5ad(h5ad_path)

    obs_back = read_df_from_h5ad(
        h5ad_path=h5ad_path,
        df_name='obs')
    pd.testing.assert_frame_equal(obs_df, obs_back)

    var_back = read_df_from_h5ad(
        h5ad_path=h5ad_path,
        df_name='var')
