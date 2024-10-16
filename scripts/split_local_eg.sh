config_dir=/home/scott.daniel/anndata_dissector/configs

parent_path=/path/to/src/h5ad/file.h5ad

output_dir=/path/to/directory/for/output/files/

python -m \
anndata_dissector.cli.split_h5ad_batch  \
--cell_config_path ${config_dir}/cell_config_10x_eg.json \
--gene_config_path ${config_dir}/gene_config_10x.json \
--tmp_dir ${TMPDIR} \
--parent_path ${parent_path} \
--output_dir ${output_dir}
