import pytest

import anndata
import pandas as pd
import numpy as np
import scipy.sparse as scipy_sparse

from anndata_dissector.utils.utils import (
    mkstemp_clean)

from anndata_dissector.modules.split_h5ad import (
    extract_h5ad)

@pytest.fixture
def n_rows_fixture():
    return 819

@pytest.fixture
def n_cols_fixture():
    return 117

def create_sparse_matrix(rng, n_rows, n_cols):
    """
    returns it densely, though
    """
    data = np.zeros(n_rows*n_cols, dtype=int)
    chosen = rng.choice(np.arange(n_rows*n_cols),
                        n_rows*n_cols//4, replace=False)
    data[chosen] = rng.integers(11, 777, len(chosen))
    data = data.reshape(n_rows, n_cols)
    return data

@pytest.fixture
def x_fixture(
        n_rows_fixture,
        n_cols_fixture):
    rng = np.random.default_rng(1623)
    return create_sparse_matrix(
        rng=rng,
        n_rows=n_rows_fixture,
        n_cols=n_cols_fixture)

@pytest.fixture
def raw_fixture(
        n_rows_fixture,
        n_cols_fixture):
    rng = np.random.default_rng(712674)
    return create_sparse_matrix(
        rng=rng,
        n_rows=n_rows_fixture,
        n_cols=n_cols_fixture)

@pytest.fixture
def parent_fixture(
        tmp_dir_fixture,
        raw_fixture,
        x_fixture):
    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='parent_',
        suffix='.h5ad')

    x_csr = scipy_sparse.csr_matrix(x_fixture)
    raw_csr = scipy_sparse.csr_matrix(raw_fixture)

    var_df = pd.DataFrame(
        data = [{'gene': f'g_{ii}'}
                for ii in range(x_fixture.shape[1])])

    a_data = anndata.AnnData(
        X=x_csr,
        var=var_df,
        layers={'rawcount': raw_csr})
 
    a_data.write_h5ad(h5ad_path)
 
    return h5ad_path


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
    np.testing.assert_array_equal(uns_actual['rows'], chosen_rows)

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
