import pytest
from itertools import product
import numpy as np
import scipy.sparse as scipy_sparse
import anndata
import os
import h5py
import tempfile
import pathlib

from anndata_dissector.utils.utils import (
    _clean_up,
    mkstemp_clean)

from anndata_dissector.utils.sparse_utils import(
    merge_csr,
    load_disjoint_csr,
    load_csr,
    merge_index_list)


@pytest.fixture
def tmp_dir_fixture(tmp_path_factory):
    tmp_path = pathlib.Path(
        tmp_path_factory.mktemp('sparse_utils_'))
    yield tmp_path
    _clean_up(tmp_path)


def test_merge_csr(tmp_dir_fixture):

    nrows = 100
    ncols = 234

    rng = np.random.default_rng(6123512)
    data = np.zeros((nrows*ncols), dtype=float)
    chosen_dex = rng.choice(np.arange(len(data)),
                            len(data)//3,
                            replace=False)
    data[chosen_dex] = rng.random(len(chosen_dex))
    data = data.reshape((nrows, ncols))

    final_csr = scipy_sparse.csr_matrix(data)

    sub0 = scipy_sparse.csr_matrix(data[:32, :])
    sub1 = scipy_sparse.csr_matrix(data[32:71, :])
    sub2 = scipy_sparse.csr_matrix(data[71:, :])

    tmp_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.h5')

    merge_csr(
         data_list=[sub0.data, sub1.data, sub2.data],
         indices_list=[sub0.indices, sub1.indices, sub2.indices],
         indptr_list=[sub0.indptr, sub1.indptr, sub2.indptr],
         tmp_path=tmp_path)

    with h5py.File(tmp_path, 'r') as src:
        merged_data = src['data'][()]
        merged_indices = src['indices'][()]
        merged_indptr = src['indptr'][()]

    np.testing.assert_allclose(merged_data, final_csr.data)
    np.testing.assert_array_equal(merged_indices, final_csr.indices)
    np.testing.assert_array_equal(merged_indptr, final_csr.indptr)


    merged_csr = scipy_sparse.csr_matrix(
        (merged_data, merged_indices, merged_indptr),
        shape=(nrows, ncols))

    result = merged_csr.todense()
    np.testing.assert_allclose(result, data)


@pytest.mark.parametrize("zero_block", (0, 1, 2))
def test_merge_csr_block_zeros(zero_block, tmp_dir_fixture):

    nrows = 100
    ncols = 234

    rng = np.random.default_rng(6123512)
    data = np.zeros((nrows*ncols), dtype=int)
    chosen_dex = rng.choice(np.arange(len(data)),
                            len(data)//3,
                            replace=False)
    data[chosen_dex] = rng.integers(3, 6000000, len(chosen_dex))
    data = data.reshape((nrows, ncols))

    if zero_block == 0:
        data[:32, :] = 0
    elif zero_block == 1:
        data[32:71, :] = 0
    elif zero_block == 2:
        data[71:, :] = 0

    final_csr = scipy_sparse.csr_matrix(data)

    sub0 = scipy_sparse.csr_matrix(data[:32, :])
    sub1 = scipy_sparse.csr_matrix(data[32:71, :])
    sub2 = scipy_sparse.csr_matrix(data[71:, :])

    tmp_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.h5')

    merge_csr(
         data_list=[sub0.data, sub1.data, sub2.data],
         indices_list=[sub0.indices, sub1.indices, sub2.indices],
         indptr_list=[sub0.indptr, sub1.indptr, sub2.indptr],
         tmp_path=tmp_path)

    with h5py.File(tmp_path, 'r') as src:
        merged_data = src['data'][()]
        merged_indices = src['indices'][()]
        merged_indptr = src['indptr'][()]

    np.testing.assert_allclose(merged_data, final_csr.data)
    np.testing.assert_array_equal(merged_indices, final_csr.indices)
    np.testing.assert_array_equal(merged_indptr, final_csr.indptr)


    merged_csr = scipy_sparse.csr_matrix(
        (merged_data, merged_indices, merged_indptr),
        shape=(nrows, ncols))

    result = merged_csr.todense()
    np.testing.assert_allclose(result, data)



def test_load_disjoint_csr(
        tmp_dir_fixture):
    nrows = 200
    ncols = 300

    tmp_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    rng = np.random.default_rng(776623)

    data = np.zeros(nrows*ncols, dtype=int)
    chosen_dex = rng.choice(np.arange(len(data)),
                            len(data)//4,
                            replace=False)

    data[chosen_dex] = rng.integers(2, 1000, len(chosen_dex))
    data = data.reshape((nrows, ncols))

    csr = scipy_sparse.csr_matrix(data)
    ann = anndata.AnnData(X=csr, dtype=int)
    ann.write_h5ad(tmp_path)

    index_list = np.unique(rng.integers(0, nrows, 45))
    rng.shuffle(index_list)
    assert len(index_list) > 2
    expected = np.zeros((len(index_list), ncols), dtype=int)

    for ct, ii in enumerate(index_list):
        expected[ct, :] = data[ii, :]

    with h5py.File(tmp_path, 'r') as written_data:
        (chunk_data,
         chunk_indices,
         chunk_indptr) = load_disjoint_csr(
                             row_index_list=index_list,
                             data=written_data['X/data'],
                             indices=written_data['X/indices'],
                             indptr=written_data['X/indptr'])

    actual = scipy_sparse.csr_matrix(
                (chunk_data, chunk_indices, chunk_indptr),
                shape=(len(index_list), ncols)).todense()

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "input_list, expected",
    (([1,2,3,8,7], [(1,4), (7,9)]),
     ([3,7,2,1,8], [(1,4), (7,9)]),
     ([0,5,9,6,10,11,17], [(0,1), (5,7), (9,12), (17,18)]),
    ))
def test_merge_index_list(input_list, expected):
    actual = merge_index_list(input_list)
    assert actual == expected


def test_load_csr(tmp_dir_fixture):
    tmp_path = pathlib.Path(
        mkstemp_clean(dir=tmp_dir_fixture, suffix='.h5ad'))

    rng = np.random.default_rng(88123)

    data = np.zeros(60000, dtype=int)
    chosen_dex = rng.choice(np.arange(len(data)),
                            len(data)//4,
                            replace=False)

    data[chosen_dex] = rng.integers(2, 1000, len(chosen_dex))
    data = data.reshape((200, 300))

    csr = scipy_sparse.csr_matrix(data)
    ann = anndata.AnnData(X=csr, dtype=int)
    ann.write_h5ad(tmp_path)

    row_spec = (119, 187)
    expected = csr[row_spec[0]:row_spec[1], :]

    with h5py.File(tmp_path, 'r') as src:
        actual = load_csr(
            row_spec=row_spec,
            data=src['X/data'],
            indices=src['X/indices'],
            indptr=src['X/indptr'])

    np.testing.assert_array_equal(
        actual[0],
        expected.data)

    np.testing.assert_array_equal(
        actual[1],
        expected.indices)

    np.testing.assert_array_equal(
        actual[2],
        expected.indptr)
