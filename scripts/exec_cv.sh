#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

dataset="gdsc2"
zscore_method="auc"
folds=10

drugs = `awk 'BEGIN {print $1}' "${homedir}/data/drugname_${dataset}.txt"`

for ont in ctg
do
	for drug in drugs
	do
		for i in {1..${folds}}
		do
			bash "${homedir}/scripts/create_cv_data.sh" $homedir $dataset $drug $folds $i
			sbatch -J "NDC_${ont}_${drug}_${i}" -o "${homedir}/logs/out_${ont}_${drug}_${i}.log" ${homedir}/scripts/cv_batch.sh $homedir $ont $dataset $drug ${zscore_method} $i
	done
done
