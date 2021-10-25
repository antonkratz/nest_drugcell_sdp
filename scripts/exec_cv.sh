#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

dataset="gdsc2"
zscore_method="auc"
folds=10

drugs=`awk 'BEGIN {print $1}' "${homedir}/data/drugname_${dataset}.txt"`

for ont in ctg
do
	for drug in "nutlin-3A" "paclitaxel" 
	do
		for ((i=1;i<=folds;i++));
		do
			bash "${homedir}/scripts/create_cv_data.sh" $homedir $dataset $drug $folds $i
			sbatch -J "NDC_${ont}_${drug}_${i}" -o "${homedir}/logs/out_${ont}_${drug}_${i}.log" ${homedir}/scripts/cv_batch.sh $homedir $ont $dataset $drug ${zscore_method} $i
        done
	done
done
