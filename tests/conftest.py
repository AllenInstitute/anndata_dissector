import pytest

import anndata
import numpy as np
import pandas as pd
import pathlib
import scipy.sparse as scipy_sparse

from anndata_dissector.utils.utils import (
    _clean_up,
    mkstemp_clean)


@pytest.fixture
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_path = pathlib.Path(
        tmp_path_factory.mktemp('general_test_'))
    yield tmp_path
    _clean_up(tmp_path)

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
