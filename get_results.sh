#!/bin/bash

dataset=$1
echo "id,algorithm,reducer,dimension,mic_f1,mac_f1,mic_cpe,mac_cpe";
for f in `ls $dataset/preds_*`; do
    trn=$dataset/fold_0/libsvm_trn.svm
    d=`basename $f`;
    d=${d#*_};
    d=${d%.*};
    tmp=${d#*-};
    dim=${tmp#*-};
    red_alg=${tmp%-*};
    alg=${d%_*};
    f1=`./accuracy3 < $f | tail -n 2 | awk 'NR==1 {macF1=$9} NR==2 {micF1=$3} END{print micF1","macF1}'`;
    cpe=`python3 compressed_predictive_efficiency.py -R 1 -f $dataset -t libsvm_trn.svm -r 128 -p $f`;
    echo "$d,$alg,$red_alg,$dim,$f1,$cpe";
done
