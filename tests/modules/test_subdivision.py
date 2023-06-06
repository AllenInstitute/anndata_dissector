import pytest

import anndata
import numpy as np
import pandas as pd
import pathlib
import scipy.sparse as scipy_sparse
import tempfile

from anndata_dissector.utils.utils import (
    mkstemp_clean,
    _clean_up)

from anndata_dissector.modules.split_h5ad import (
    subdivide_h5ad)


@pytest.fixture
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('subdivision'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def x_fixture():
    rng = np.random.default_rng(2231244)
    n_rows = 114
    n_cols = 212
    data = np.zeros(n_rows*n_cols, dtype=float)
    chosen = rng.choice(
            n_rows*n_cols,
            n_rows*n_cols//3,
            replace=False)
    data[chosen] = rng.random(len(chosen))
    data = data.reshape((n_rows, n_cols))
    return data


@pytest.fixture
def parent_obs_data_fixture(x_fixture):
    rng = np.random.default_rng(88122)
    n_cells = x_fixture.shape[0]
    data = []
    for ii in range(n_cells):
        this = {'cell_id': f'cell_{ii}',
                'garbage': rng.integers(0, 1111),
                'sum': x_fixture[ii, :].sum()}
        data.append(this)
    return data

@pytest.fixture
def parent_var_data_fixture(x_fixture):
    rng = np.random.default_rng(61232)
    n_genes = x_fixture.shape[1]
    data = []
    for ii in range(n_genes):
        this = {'gene_id': f'gene_{ii}',
                'junk': rng.integers(2222, 5555),
                'sum': x_fixture[:,  ii].sum()}
        data.append(this)
    return data


@pytest.fixture
def parent_h5ad_fixture(
        x_fixture,
        parent_var_data_fixture,
        parent_obs_data_fixture,
        tmp_dir_fixture):

    tmp_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    obs = pd.DataFrame(parent_obs_data_fixture).set_index('cell_id')
    var = pd.DataFrame(parent_var_data_fixture).set_index('gene_id')
    x_matrix = scipy_sparse.csr_matrix(x_fixture)
    a_data = anndata.AnnData(
        X=x_matrix,
        obs=obs,
        var=var)
    a_data.write_h5ad(tmp_path)
    return pathlib.Path(tmp_path)


@pytest.mark.parametrize(
    "n_sub", [3, 4, 5])
def test_doing_subdivision(
        x_fixture,
        parent_obs_data_fixture,
        parent_var_data_fixture,
        parent_h5ad_fixture,
        tmp_dir_fixture,
        n_sub):

    output_dir = pathlib.Path(
        tempfile.mkdtemp(dir=tmp_dir_fixture))

    subdivide_h5ad(
        parent_path=parent_h5ad_fixture,
        number_of_subdivisions=n_sub,
        output_dir=output_dir,
        clobber=False)

    output_contents = [n for n in output_dir.iterdir()]

    assert len(output_contents) == n_sub

    for pth in output_contents:
        assert pth.is_file()
        print(pth)

    expected_var = pd.DataFrame(
        parent_var_data_fixture).set_index('gene_id')

    n_cells = len(parent_obs_data_fixture)
    n_per = np.ceil(n_cells/n_sub).astype(int)

    row_ct = 0
    for ii in range(1, n_sub+1, 1):
        expected_name = parent_h5ad_fixture.name.replace(
            '.h5ad',
            f'-{ii}.h5ad')
        expected_path = output_dir / expected_name
        assert expected_path.is_file()
        actual_anndata = anndata.read_h5ad(expected_path, backed='r')
        row_ct += actual_anndata.X.shape[0]

        pd.testing.assert_frame_equal(actual_anndata.var, expected_var)

        expected_obs = parent_obs_data_fixture[
            (ii-1)*n_per:ii*n_per]
        expected_obs = pd.DataFrame(expected_obs).set_index('cell_id')
        pd.testing.assert_frame_equal(actual_anndata.obs, expected_obs)

        expected_x = x_fixture[
            (ii-1)*n_per:ii*n_per, :]
        actual_x = actual_anndata.X[()].toarray()
        np.testing.assert_allclose(
            actual_x,
            expected_x,
            atol=0.0,
            rtol=1.0e-6)

    assert row_ct == x_fixture.shape[0]
