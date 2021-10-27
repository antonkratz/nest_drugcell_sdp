#!/bin/bash
#SBATCH --job-name=NeST_DrugCell
#SBATCH --output=out.log
#SBATCH --partition=nrnb-gpu
#SBATCH --account=nrnb-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --dependency=singleton

bash "${1}/scripts/cv_train.sh" $1 $2 $3 $4 $5 $6
bash "${1}/scripts/cv_test.sh" $1 $2 $3 $4 $5 $6
