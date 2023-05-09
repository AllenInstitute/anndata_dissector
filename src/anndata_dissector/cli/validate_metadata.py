import anndata
import argparse
import copy
import json
import numpy as np
import pathlib


def validate_metadata(
        src_path,
        cell_config,
        gene_config):

    test = anndata.read_h5ad(src_path, backed='r')

    assert len(cell_config) == len(test.uns['parent_rows'])
    expected_records = []
    parent_rows = test.uns['parent_rows']
    for ii in range(len(cell_config)):
        cell = copy.deepcopy(cell_config[ii])
        assert cell['row_idx'] == parent_rows[ii]
        cell.pop('row_idx')
        expected_records.append(cell)

    obs = test.obs
    obs_records = obs.to_dict(orient='records')
    assert len(obs_records) == len(cell_config)
    for ii in range(len(obs_records)):
        this_record = copy.deepcopy(obs_records[ii])
        this_record[obs.index.name] = obs.index.values[ii]
        assert this_record == expected_records[ii]
    print(f"validated {len(obs_records)} cell records")

    var = test.var
    var_records = var.to_dict(orient='records')
    assert len(var_records) == len(gene_config)
    for ii in range(len(var_records)):
        this_record = copy.deepcopy(var_records[ii])
        this_record[var.index.name] = var.index.values[ii]
        assert this_record == gene_config[ii]

    print(f'validated {len(gene_config)} gene records')

    print('all done')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default=None)
    parser.add_argument('--cell_config_path', type=str, default=None)
    parser.add_argument('--gene_config_path', type=str, default=None)
    args = parser.parse_args()

    cell_config = json.load(open(args.cell_config_path, 'rb'))
    gene_config = json.load(open(args.gene_config_path, 'rb'))
    src_path = pathlib.Path(args.src_path)

    config_key = src_path.name.replace(f"-{src_path.name.split('-')[-1]}", '')
    cell_config = cell_config[config_key]

    validate_metadata(
        src_path=src_path,
        cell_config=cell_config,
        gene_config=gene_config)


if __name__ == "__main__":
    main()
