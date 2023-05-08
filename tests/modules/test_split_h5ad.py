import pytest

import anndata
import pandas as pd
import numpy as np

from anndata_dissector.utils.utils import (
    mkstemp_clean)

from anndata_dissector.modules.split_h5ad import (
    extract_h5ad)


@pytest.mark.parametrize(
    'layer', ['X', 'rawcount'])
def test_extract_h5ad(
        parent_fixture,
        x_fixture,
        raw_fixture,
        layer,
        n_rows_fixture,
        n_cols_fixture,
        tmp_dir_fixture):
    if layer == 'X':
        baseline = x_fixture
    else:
        baseline = raw_fixture

    rng = np.random.default_rng(1231)
    chosen_rows = rng.choice(np.arange(n_rows_fixture, dtype=int),
                             15, replace=False)
    assert not np.array_equal(chosen_rows, np.sort(chosen_rows))

    cell_config = [
       {'row_idx': ii,
        'new_idx': f'row_{ct}',
        'other_metadata': 'b'*ct}
       for ct, ii in enumerate(chosen_rows)]

    gene_config = [
       {'gene_idx': f'g_{ii}',
        'other': 'y'*ii}
       for ii in range(n_cols_fixture)]

    out_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    extract_h5ad(
        parent_path=parent_fixture,
        cell_config=cell_config,
        gene_config=gene_config,
        layer=layer,
        output_path=out_path,
        obs_index_column='new_idx',
        var_index_column='gene_idx',
        metadata={'norm': 'silly'},
        clobber=True)

    actual = anndata.read_h5ad(out_path, backed='r')
    x_actual = actual.X[()].toarray()
    np.testing.assert_array_equal(
        x_actual,
        baseline[chosen_rows, :])

    uns_actual = actual.uns
    assert uns_actual['norm'] == 'silly'
    assert uns_actual['parent'] == str(parent_fixture)
    if layer == 'X':
        expected_layer = 'X'
    else:
        expected_layer = f'layers/{layer}'
    assert uns_actual['parent_layer'] == expected_layer
    np.testing.assert_array_equal(
        uns_actual['parent_rows'],
        chosen_rows)

    obs_data = []
    for cell in cell_config:
        cell.pop('row_idx')
        obs_data.append(cell)
    obs_df = pd.DataFrame(obs_data)
    obs_df = obs_df.set_index('new_idx')
    pd.testing.assert_frame_equal(actual.obs, obs_df)

    var_df = pd.DataFrame(gene_config)
    var_df = var_df.set_index('gene_idx')
    pd.testing.assert_frame_equal(actual.var, var_df)
