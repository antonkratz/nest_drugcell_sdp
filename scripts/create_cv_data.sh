#!/bin/bash

homedir=$1
dataset=$2
drug=$3
folds=$4
i=$5
class=$6

if [ $class == "" ]; then
	echo "Processing regression task"

	datadir="${homedir}/data/training_files"
	dataFile="${datadir}/train_${dataset}_${drug}.txt"
	trainFile="train_${dataset}_${drug}.txt"
	testFile="test_${dataset}_${drug}.txt"

	datadir="${homedir}/data/training_files"
	dataFile="${datadir}/train_${dataset}_${drug}.txt"
	trainFile="train_${dataset}_${drug}.txt"
	testFile="test_${dataset}_${drug}.txt"

	lc=`cat ${dataFile} | wc -l`

	min=$(( ($lc * (${i} - 1)) / ${folds} + 1 ))
	max=$(( ($lc * ${i}) / ${folds} ))

	sed -n "${min},${max}p" $dataFile > "${datadir}/${i}_${testFile}"
	if [[ $min > 1 ]]
	then
		min=$(( $min - 1 ))
		sed -n "1,${min}p" $dataFile >> "${datadir}/${i}_${trainFile}"
	fi
	if [[ $max < $lc ]]
	then
		max=$(( $max + 1))
		sed -n "${max},${lc}p" $dataFile >> "${datadir}/${i}_${trainFile}"
	fi

elif [ $class == "ord" ]; then
    echo "Processing ordinal classification task"
		echo "Processing regression task"
	datadir="${homedir}/data/class/training_files"
	dataFile="${datadir}/${dataset}.${drug}.${class}.txt"
	trainFile="train.${dataset}.${drug}.${class}.txt"
	testFile="test.${dataset}.${drug}.${class}.txt"

	datadir="${homedir}/data/class/training_files"
	dataFile="${datadir}/train_${dataset}_${drug}.txt"
	trainFile="train_${dataset}_${drug}.txt"
	testFile="test_${dataset}_${drug}.txt"

	lc=`cat ${dataFile} | wc -l`

	min=$(( ($lc * (${i} - 1)) / ${folds} + 1 ))
	max=$(( ($lc * ${i}) / ${folds} ))

	sed -n "${min},${max}p" $dataFile > "${datadir}/${i}_${testFile}"
	if [[ $min > 1 ]]
	then
		min=$(( $min - 1 ))
		sed -n "1,${min}p" $dataFile >> "${datadir}/${i}_${trainFile}"
	fi
	if [[ $max < $lc ]]
	then
		max=$(( $max + 1))
		sed -n "${max},${lc}p" $dataFile >> "${datadir}/${i}_${trainFile}"
	fi

elif [ $class == "onehot" ]; then
    echo "Processing onehot classification task"

fi
