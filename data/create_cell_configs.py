import json

raw_path = "raw/extracted_wb10x_cell_info.csv"

config = dict()

with open(raw_path, "r") as in_file:
    header = in_file.readline().strip().split(',')
    header_to_idx = {
        n:ii for ii, n in enumerate(header)}
    for line in in_file:
        params = line.strip().split(',')
        if len(params) != len(header):
            raise RuntimeError("line mismatch")
        feature = params[header_to_idx['feature_matrix_label']]
        if feature not in config:
            config[feature] = []
        current_config = config[feature]

        this_cell = {
            'row_idx': int(params[header_to_idx['source_row_index']]),
            'cell_label': params[header_to_idx['cell_label']],
            'cell_barcode': params[header_to_idx['cell_barcode']],
            'library_label': params[header_to_idx['library_label']],
            'anatomical_division_label': params[
                    header_to_idx['anatomical_division_label']]
        }
        current_config.append(this_cell)

with open('configs/cell_config.json', 'w') as out_file:
    out_file.write(json.dumps(config, indent=2))
