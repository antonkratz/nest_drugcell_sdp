#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

dataset="gdsc2"
zscore_method="auc"

drugs = `awk 'BEGIN {FS="\t"} {print $2}' "${homedir}/data/drug2ind_${dataset}.txt"`

for ont in ctg
do
	for drug in drugs
	do
		sbatch -J "NDC_${ont}_${drug}" -o "${homedir}/logs/out_${ont}_${drug}.log" ${homedir}/scripts/batch.sh $homedir $ont $dataset $drug ${zscore_method}
	done
done
