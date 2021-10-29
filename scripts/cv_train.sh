#!/bin/bash
homedir=$1
zscore_method=$5
class=$7

echo $1 $2 $3 $4 $5 $6 $7

gene2idfile="${homedir}/data/gene2ind_${2}_${3}.txt"
cell2idfile="${homedir}/data/cell2ind_${3}.txt"
ontfile="${homedir}/data/ontology_${2}_${3}.txt"
mutationfile="${homedir}/data/cell2mutation_${2}_${3}.txt"

if [ $class == "" ]; then
	traindatafile="${homedir}/data/training_files/${6}_train_${3}_${4}.txt"
	modeldir="${homedir}/models/model_${3}_${4}_${5}_${6}"
elif [ $class == "ord" ]; then
	traindatafile="${homedir}/data/class/training_files/${6}_train.${3}.${4}.${class}.txt"
	modeldir="${homedir}/models/model_${3}_${4}_${5}_${6}_${class}"
elif [ $class == "onehot" ]; then
	traindatafile="${homedir}/data/class/training_files/${6}_train.${3}.${4}.${class}.txt"
	modeldir="${homedir}/models/model_${3}_${4}_${5}_${6}_${class}"
fi

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

if [ $class == "" ]; then
	python -u $pyScript -onto $ontfile -gene2id $gene2idfile -cell2id $cell2idfile \
	-train $traindatafile -genotype $mutationfile -std $stdfile -model $modeldir \
	-genotype_hiddens 2 -lr 0.0005 -wd 0.0001 -alpha 0.2 -cuda $cudaid -epoch 50 \
	-batchsize 100 -optimize 2 -zscore_method $zscore_method > "${modeldir}/train.log" 
elif [ $class == "ord" ]; then
	python -u $pyScript -onto $ontfile -gene2id $gene2idfile -cell2id $cell2idfile \
	-train $traindatafile -genotype $mutationfile -std $stdfile -model $modeldir \
	-genotype_hiddens 2 -lr 0.0005 -wd 0.0001 -alpha 0.2 -cuda $cudaid -epoch 50 \
	-batchsize 100 -optimize 0 -n_classes 3 > "${modeldir}/train.log"
elif [ $class == "onehot" ]; then
	python -u $pyScript -onto $ontfile -gene2id $gene2idfile -cell2id $cell2idfile \
	-train $traindatafile -genotype $mutationfile -std $stdfile -model $modeldir \
	-genotype_hiddens 2 -lr 0.0005 -wd 0.0001 -alpha 0.2 -cuda $cudaid -epoch 50 \
	-batchsize 100 -optimize 2 -n_classes 3 > "${modeldir}/train.log"
fi

