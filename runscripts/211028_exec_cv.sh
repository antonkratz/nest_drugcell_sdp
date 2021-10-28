#!/bin/bash

homedir="/cellar/users/e5silva/Software/nest_drugcell"

dataset="gdsc2"
zscore_method=""
folds=10
class="ord"

drugs = `awk 'BEGIN {print $1}' "${homedir}/data/drugname_${dataset}.txt"`

for ont in ctg
do
	for drug in "palbociclib" "nutlin-3A" "paclitaxel" "trametinib" "dabrafenib" "vincristine" "sorafenib" "docetaxel" "epirubicin" "cediranib"
	do
		for i in {1..${folds}}
		do
			bash "${homedir}/scripts/create_cv_data.sh" $homedir $dataset $drug $folds $i $class
			sbatch -J "NDC_${ont}_${drug}_${i}" -o "${homedir}/logs/out_${ont}_${drug}_${i}.log" ${homedir}/scripts/cv_batch.sh $homedir $ont $dataset $drug ${zscore_method} $i
	done
done
