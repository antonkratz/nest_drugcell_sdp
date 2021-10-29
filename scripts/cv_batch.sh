#!/bin/bash
#SBATCH --job-name=NeST_DrugCell
#SBATCH --output=slurm-%A.out
#SBATCH --partition=nrnb-gpu
#SBATCH --account=nrnb-gpu
#SBATCH --mem=16G
#SBATCH --dependency=singleton

bash "${1}/scripts/cv_train.sh" $1 $2 $3 $4 $5 $6 $7
bash "${1}/scripts/cv_test.sh" $1 $2 $3 $4 $5 $6 $7
