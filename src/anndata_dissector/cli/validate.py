import anndata
import argparse
import numpy as np


def validate_h5ad(
        src_path):

    test = anndata.read_h5ad(src_path, backed='r')
    parent_path = test.uns['parent']
    parent = anndata.read_h5ad(parent_path, backed='r')

    if test.uns['parent_layer'] == 'X':
        baseline = parent.chunk_X(test.uns['parent_rows'])
    else:
        _layer = test.uns['parent_layer'].replace('layers/','')
        baseline = parent.layers[_layer][test.uns['parent_rows'], :]
        baseline = baseline.toarray()

    np.testing.assert_allclose(
        test.X[()].toarray(),
        baseline)

    assert test.obs.index.name == 'cell_label'
    assert test.var.index.name == 'gene_identifier'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default=None)
    args = parser.parse_args()

    validate_h5ad(args.src_path)


if __name__ == "__main__":
    main()
