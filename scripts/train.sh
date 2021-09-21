#!/bin/bash
homedir=$1
gene2idfile="${homedir}/data/gene2ind_${2}.txt"
cell2idfile="${homedir}/data/cell2ind_cg.txt"
drug2idfile="${homedir}/data/drug2ind_cg.txt"
ontfile="${homedir}/data/ontology_${2}.txt"
mutationfile="${homedir}/data/cell2mutation_${2}.txt"
drugfile="${homedir}/data/drug2fingerprint_cg.txt"
traindatafile="${homedir}/data/drugcell_train_cg.txt"
valdatafile="${homedir}/data/drugcell_val_cg.txt"
zscore_method=$3

modeldir="${homedir}/model_${2}"
if [ -d $modeldir ]
then
	rm -rf $modeldir
fi
mkdir -p $modeldir

stdfile="${modeldir}/std.txt"

cudaid=0

pyScript="${homedir}/src/train_drugcell.py"

source activate cuda11_env

python -u $pyScript -onto $ontfile -gene2id $gene2idfile -drug2id $drug2idfile \
	-cell2id $cell2idfile -train $traindatafile -val $valdatafile -genotype $mutationfile -std $stdfile \
	-fingerprint $drugfile -genotype_hiddens 6 -drug_hiddens '100,50,6' -final_hiddens 6 -lr 0.05 -wd 0.001 \
	-model $modeldir -cuda $cudaid -batchsize 20000 -epoch 50 -optimize 0 -zscore_method $zscore_method > "${modeldir}/train.log"
