#!/bin/bash

homedir="/cellar/users/e5silva/Software/nest_drugcell"

dataset="gdsc2"
zscore_method="none"
folds=10
class="ord"

drugs=`awk 'BEGIN {print $1}' "${homedir}/data/drugname_${dataset}.txt"`

for ont in ctg
do
	# for drug in "palbociclib" "nutlin-3a" "paclitaxel" "trametinib" "dabrafenib" "vincristine" "sorafenib" "docetaxel" "epirubicin" "cediranib"
	for drug in "palbociclib"
	do
		for ((i=1;i<=folds;i++));
		do
			# bash "${homedir}/scripts/create_cv_data.sh" $homedir $dataset $drug $folds $i $class
			sbatch -J "NDC_${ont}_${drug}_${i}" -o "${homedir}/logs/out_${ont}_${drug}_${i}.log" ${homedir}/scripts/cv_batch.sh $homedir $ont $dataset $drug ${zscore_method} $i $class
		done
	done
done