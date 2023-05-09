import h5py
import numpy as np
import pathlib
import tempfile

from anndata_dissector.utils.utils import (
    _clean_up,
    mkstemp_clean)


def load_disjoint_csr(
        row_index_list,
        data,
        indices,
        indptr,
        tmp_dir=None):
    """
    Load a csr matrix from a not necessarily contiguous
    set of row indexes.
    """
    tmp_dir = pathlib.Path(
        tempfile.mkdtemp(dir=tmp_dir,
                         prefix='csr_'))


    row_index_list = np.array(row_index_list)

    (row_index_list,
     inverse_argsort,
     merged_data,
     merged_indices,
     merged_indptr) = _load_csr_naive(
                             row_index_list=row_index_list,
                             data=data,
                             indices=indices,
                             indptr=indptr,
                             tmp_dir=tmp_dir)

    # undo sorting
    final_data = np.zeros(merged_data.shape, dtype=merged_data.dtype)
    final_indices = np.zeros(merged_indices.shape, dtype=merged_indices.dtype)
    final_indptr = np.zeros(merged_indptr.shape, dtype=merged_indptr.dtype)

    data_ct = 0
    for ii in range(len(row_index_list)):
        new_position = inverse_argsort[ii]
        indptr0 = merged_indptr[new_position]
        indptr1 = merged_indptr[new_position+1]
        n = indptr1-indptr0
        final_data[data_ct:data_ct+n] = merged_data[indptr0:indptr1]
        final_indices[data_ct:data_ct+n] = merged_indices[indptr0:indptr1]
        final_indptr[ii] = data_ct
        data_ct += n
    final_indptr[-1] = len(final_data)

    _clean_up(tmp_dir)

    return final_data, final_indices, final_indptr


def _load_csr_naive(
        row_index_list,
        data,
        indices,
        indptr,
        tmp_dir):

    sorted_dex = np.argsort(row_index_list)
    inverse_argsort = {sorted_dex[ii]: ii for ii in range(len(sorted_dex))}

    row_index_list = row_index_list[sorted_dex]

    row_chunk_list = merge_index_list(row_index_list)
    data_list = []
    indices_list = []
    indptr_list = []
    for row_chunk in row_chunk_list:
        (this_data,
         this_indices,
         this_indptr) = _load_csr(
                             row_spec=row_chunk,
                             data=data,
                             indices=indices,
                             indptr=indptr)

        data_list.append(this_data)
        indices_list.append(this_indices)
        indptr_list.append(this_indptr)

    (merged_data,
     merged_indices,
     merged_indptr) = merge_csr(
                         data_list=data_list,
                         indices_list=indices_list,
                         indptr_list=indptr_list,
                         tmp_dir=tmp_dir)

    return (row_index_list,
            inverse_argsort,
            merged_data,
            merged_indices,
            merged_indptr)


def _load_csr(
        row_spec,
        data,
        indices,
        indptr):
    """
    Load a subset of rows from a matrix stored as a
    csr matrix (probably in zarr format).

    Parameters
    ----------
    row_spec:
        A tuple of the form (row_min, row_max)

    data:
        The data matrix (as in scipy.sparse.csr_matrix().data)

    indices:
        The indices matrix (as in scipy.sparse.csr_matrix().indices)

    indptr:
        The indptr matrix (as in scipy.sparse.csr_matrix().indptr)

    Returns
    -------
    The appropriate slices of data, indices, indptr
    """
    these_ptrs = indptr[row_spec[0]:row_spec[1]+1]
    index0 = these_ptrs[0]
    index1 = these_ptrs[-1]
    these_indices = indices[index0:index1]
    this_data = data[index0:index1]
    return this_data, these_indices, these_ptrs-these_ptrs.min()


def merge_csr(
        data_list,
        indices_list,
        indptr_list,
        tmp_dir):
    """
    Merge multiple CSR matrices into one

    Parameters
    ----------
    data_list:
        List of the distinct 'data' arrays from
        the CSR matrices

    indices_list:
        List of the distinct 'indices' arrays from
        the CSR matrices

    indptr_list:
        List of the distinct 'indptr' arrays from
        the CSR matrices

    Returns
    -------
    data:
        merged 'data' array for the final CSR matrix

    indices:
        merged 'indices' array for the final CSR matrix

    indptr:
        merged 'indptr' array for the final CSR matrix

    Note
    -----
    will pop elements from data_list, indices_list, and indptr_list
    so that those arguments get modified in place.
    """
    n_data = 0
    for d in data_list:
        n_data += len(d)
    n_indptr = 0
    for i in indptr_list:
        n_indptr += len(i)-1
    n_indptr += 1

    tmp_path = mkstemp_clean(dir=tmp_dir, suffix='.h5')
    chunks = (min(n_data, 10000), )
    with h5py.File(tmp_path, 'a') as out_file:
        data = out_file.create_dataset(
            'data',
            shape=(n_data,),
            dtype=data_list[0].dtype,
            chunks=chunks)
        indices = out_file.create_dataset(
            'indices',
            shape=(n_data,),
            dtype=int,
            chunks=chunks)
        indptr = out_file.create_dataset(
            'indptr',
            shape=(n_indptr,),
            dtype=int,
            chunks=(min(n_indptr, 1000),))

        i0 = 0
        ptr0 = 0
        for ii in range(len(data_list)):
            this_data = data_list[ii]
            this_indices = indices_list[ii]
            this_indptr = indptr_list[ii]

            data_list[ii] = None
            indices_list[ii] = None
            indptr_list[ii] = None

            (idx1,
             ptr1) = _merge_csr_chunk(
                     data_in=this_data,
                     indices_in=this_indices,
                     indptr_in=this_indptr,
                     data=data,
                     indices=indices,
                     indptr=indptr,
                     idx0=i0,
                     ptr0=ptr0)

            i0 = idx1
            ptr0 = ptr1

        indptr[-1] = n_data
        data = data[()]
        indices = indices[()]
        indptr = indptr[()]

    assert data.shape == (n_data, )
    return data, indices, indptr


def _merge_csr_chunk(
        data_in,
        indices_in,
        indptr_in,
        data,
        indices,
        indptr,
        idx0,
        ptr0):
    idx1 = idx0 + len(data_in)
    data[idx0:idx1] = data_in
    indices[idx0: idx1] = indices_in
    ptr1 = ptr0 + len(indptr_in)-1
    indptr[ptr0:ptr1] = indptr_in[:-1] + idx0

    return idx1, ptr1


def merge_index_list(
        index_list):
    """
    Take a list of integers, merge those that can be merged into
    (min, max) ranges for slicing array. Return as a list of those
    tuples. Note that max will be 1 greater than any value in the array
    because of the way array slicing works.
    """
    index_list = np.unique(index_list)
    diff_list = np.diff(index_list)
    breaks = np.where(diff_list > 1)[0]
    result = []
    min_dex = 0
    for max_dex in breaks:
        result.append((index_list[min_dex],
                       index_list[max_dex]+1))
        min_dex = max_dex+1
    result.append((index_list[min_dex], index_list[-1]+1))
    return result
