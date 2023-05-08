import anndata
import json
import numpy as np
import pathlib
import tempfile

from anndata_dissector.cli.split_h5ad_batch import (
    split_h5ad_batch)


def test_batch_extraction(
    parent_fixture,
    n_rows_fixture,
    n_cols_fixture,
    tmp_dir_fixture,
    x_fixture,
    raw_fixture):

    gene_config = [
       {'gene_idx': f'g_{ii}',
        'other': 'y'*ii}
       for ii in range(n_cols_fixture)]

    cell_config = dict()
    rng = np.random.default_rng(75231)
    row_lookup = dict()
    for name in 'abcde':
        chosen_rows = rng.choice(
            np.arange(n_rows_fixture, dtype=int),
            rng.integers(15, 26),
            replace=False)

        row_lookup[name] = chosen_rows

        this_config = [
            {'row_idx': ii,
             'new_idx': f'{name}_{ii}',
             'other_metadata': f'{name}_{ii**2:d}'}
            for ii in chosen_rows]
        cell_config[name] = this_config

    output_dir = tempfile.mkdtemp(
        dir=tmp_dir_fixture,
        prefix='batch_output_')

    split_h5ad_batch(
        parent_path=parent_fixture,
        cell_config_lookup=cell_config,
        gene_config=gene_config,
        obs_index_column='new_idx',
        var_index_column='gene_idx',
        output_dir=output_dir,
        tmp_dir=tmp_dir_fixture,
        clobber=False)

    output_dir = pathlib.Path(output_dir)

    ct = 0
    for name in cell_config:
        for norm in ('log2', 'raw'):
            expected_path = output_dir / f'{name}-{norm}.h5ad'
            assert expected_path.is_file()
            ct += 1
            actual = anndata.read_h5ad(expected_path, backed='r')

            expected_rows = row_lookup[name]
            if norm == 'log2':
                expected = x_fixture[
                    expected_rows, :]
            else:
                expected = raw_fixture[
                    expected_rows, :]

            np.testing.assert_allclose(
                actual.X[()].toarray(),
                expected,
                atol=0.0,
                rtol=1.0e-6)

            assert actual.uns['parent'] == parent_fixture
            np.testing.assert_array_equal(
                actual.uns['parent_rows'], expected_rows)
            if norm == 'log2':
                assert actual.uns['normalization'] == 'log2(CPM+1)'
            else:
                assert actual.uns['normalization'] == 'raw'

            assert actual.obs.index.name == 'new_idx'
            assert actual.var.index.name == 'gene_idx'


    assert ct == 10
