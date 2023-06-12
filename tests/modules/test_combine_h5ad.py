import pytest

import anndata
import numpy as np
import pandas as pd
import pathlib
import scipy.sparse as scipy_sparse

from anndata_dissector.utils.utils import (
    mkstemp_clean,
    _clean_up)

from anndata_dissector.modules.combine_h5ad import (
    combine_h5ad_row_wise)


@pytest.fixture
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('combine_h5ad_test_'))
    yield tmp_dir
    _clean_up(tmp_dir)

@pytest.fixture
def x_parent_fixture():
    rng = np.random.default_rng(77661122)
    n_rows = 249
    n_cols = 179
    n_tot = n_rows*n_cols
    data = np.zeros(n_tot, dtype=float)
    chosen = rng.choice(np.arange(n_tot), n_tot//5, replace=False)
    data[chosen] = rng.random(len(chosen))
    return data.reshape(n_rows, n_cols)


@pytest.fixture
def obs_parent_fixture(x_parent_fixture):
    rng = np.random.default_rng(98123)
    n_rows = x_parent_fixture.shape[0]
    obs_data = []
    for ii in range(n_rows):
        this = {'row_idx': f'row_{ii}',
                'garbage': rng.integers(9, 19999),
                'junk': rng.integers(88,200000000)}
        obs_data.append(this)
    return obs_data


@pytest.fixture
def var_parent_fixture(x_parent_fixture):
    rng = np.random.default_rng(7612322)
    n_cols = x_parent_fixture.shape[1]
    var_data = []
    for ii in range(n_cols):
        this = {'col_idx': f'col_{ii}',
                'silly': rng.integers(1000000,2000000),
                'hilarious': rng.integers(4000000,10000000)}
        var_data.append(this)
    return var_data


@pytest.fixture
def h5ad_list_fixture(
        x_parent_fixture,
        obs_parent_fixture,
        var_parent_fixture,
        tmp_dir_fixture):
    """
    Break monolithic data into several
    h5ad files that can be joined
    """

    var = pd.DataFrame(var_parent_fixture).set_index('col_idx')

    rng = np.random.default_rng(2010312)
    n_rows = x_parent_fixture.shape[0]
    n_per_max = n_rows // 4
    n_per_min = n_rows // 6

    assert n_per_min < n_per_max
    i0 = 0
    i1 = 0
    h5ad_path_list = []
    while i1 < n_rows:
        n_per = rng.integers(n_per_min, n_per_max)
        i1 = min(n_rows, i0+n_per)
        this_obs = pd.DataFrame(obs_parent_fixture[i0:i1])
        this_obs = this_obs.set_index('row_idx')
        this_csr = scipy_sparse.csr_matrix(
            x_parent_fixture[i0:i1, :])
        a_data = anndata.AnnData(
            X=this_csr,
            obs=this_obs,
            var=var,
            uns={'hello': 'there',
                 'number': i0,
                 'other_number': i1})
        h5ad_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5ad')
        a_data.write_h5ad(h5ad_path)
        h5ad_path_list.append(h5ad_path)
        i0 = i1

    return h5ad_path_list


def test_combine_h5ad(
        h5ad_list_fixture,
        x_parent_fixture,
        obs_parent_fixture,
        var_parent_fixture,
        tmp_dir_fixture):

    assert len(h5ad_list_fixture) > 3

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    combine_h5ad_row_wise(
        h5ad_path_list=h5ad_list_fixture,
        output_path=output_path,
        tmp_dir=tmp_dir_fixture)

    actual = anndata.read_h5ad(output_path, backed='r')
    expected_var = pd.DataFrame(var_parent_fixture).set_index('col_idx')
    pd.testing.assert_frame_equal(actual.var, expected_var)

    expected_obs = pd.DataFrame(obs_parent_fixture).set_index('row_idx')

    actual_x = actual.X[()].toarray()
    np.testing.assert_allclose(
        actual_x,
        x_parent_fixture,
        atol=0.0,
        rtol=1.0e-6)

    assert actual.uns == {'hello': 'there'}
