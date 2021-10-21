#!/bin/bash
homedir=$1
gene2idfile="${homedir}/data/gene2ind_${2}_${3}.txt"
cell2idfile="${homedir}/data/cell2ind_${2}.txt"
ontfile="${homedir}/data/ontology_${2}_${3}.txt"
mutationfile="${homedir}/data/cell2mutation_${2}_${3}.txt"
traindatafile="${homedir}/data/training_files/train_${3}_${4}.txt"
zscore_method=$5

modeldir="${homedir}/model_${3}_${4}_${5}"
if [ -d $modeldir ]
then
	rm -rf $modeldir
fi
mkdir -p $modeldir

stdfile="${modeldir}/std.txt"
resultfile="${modeldir}/predict"

cudaid=0

pyScript="${homedir}/src/train_drugcell.py"

source activate cuda11_env

python -u $pyScript -onto $ontfile -gene2id $gene2idfile -cell2id $cell2idfile \
	-train $traindatafile -genotype $mutationfile -std $stdfile -result $resultfile \
	-model $modeldir -genotype_hiddens 2 -final_hiddens 6 -lr 0.0005 -wd 0.0001 -alpha 0.2 \
	-cuda $cudaid -batchsize 100 -epoch 30 -optimize 0 -zscore_method $zscore_method > "${modeldir}/train.log"
