#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

dataset="gdsc2"
zscore_method="auc"

drugs = `awk 'BEGIN {FS="\t"} {print $2}' "${homedir}/data/drug2ind_${dataset}.txt"`

for ont in ctg
do
	do drug in drugs
		sbatch --job-name "NDC_${ont}_$drug_$1" --output "${homedir}/logs/out_${ont}_$drug_$1.log" ${homedir}/scripts/batch.sh $homedir $ont $dataset $drug ${zscore_method}
	#sbatch --job-name "NDC_${ontology}" --output "out_${ontology}.log" ${homedir}/scripts/rlipp_slurm.sh $homedir $ontology
done
