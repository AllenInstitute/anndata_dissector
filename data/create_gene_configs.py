import json

config = []
with open("raw/extracted_wb10x_gene_identifiers.csv", "r") as in_file:
    header = in_file.readline().strip().split(',')
    header_to_idx = {
       n:ii for ii, n in enumerate(header)}

    for line in in_file:
        params = line.strip().split(',')
        this = {
            'row_idx': int(params[header_to_idx['']]),
            'gene_identifier': params[header_to_idx['gene_identifer']],
            'gene_symbol': params[header_to_idx['gene_symbol']]}
        config.append(this)
with open('configs/gene_config.json', 'w') as out_file:
    out_file.write(json.dumps(config, indent=2))
