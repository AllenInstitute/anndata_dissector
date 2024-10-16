#!/bin/bash
#SBATCH --job-name=10x_split
#SBATCH --mail-type=NONE
#SBATCH --cpus-per-task=1
#SBATCH --mem=128gb
#SBATCH --time=12:00:00
#SBATCH --output=/home/scott.daniel/anndata_dissector/scripts/output/split_10x_%A.txt
#SBATCH --partition celltypes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --tmp=500gb
#SBATCH --exclude=n241

source /home/scott.daniel/single_core.sh
source /home/scott.daniel/informatics_conda3_cmd.sh
conda activate anndata_dissector

config_dir=/home/scott.daniel/anndata_dissector/configs

parent_path=/path/to/src/h5ad/file.had

output_dir=/path/to/directory/for/output/files/

python -m \
anndata_dissector.cli.split_h5ad_batch  \
--cell_config_path ${config_dir}/cell_config_10x_eg.json \
--gene_config_path ${config_dir}/gene_config_10x.json \
--tmp_dir ${TMPDIR} \
--parent_path ${parent_path} \
--output_dir ${output_dir} 
