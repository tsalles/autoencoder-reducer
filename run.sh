#!/bin/bash

dataset=$1
num_folds=$2

rm $dataset/preds_bl.out;
rm $dataset/preds_ae.out;

for i in `seq 0 $((num_folds-1))`; do
  echo "Round $i"
  python3 model.py -t $dataset/${num_folds}fold/train$i -T $dataset/${num_folds}fold/test$i -o $dataset/preds_bl.out -c 128;
  python3 model.py -t $dataset/${num_folds}fold/train$i -T $dataset/${num_folds}fold/test$i -a 256 128 -b 64 -c 2048 1024 --autoencoder -o $dataset/preds_ae.out;
done
