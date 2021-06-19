#!/bin/bash

dataset=$1
num_folds=$2

$liblinear_train="/home/tsalles/Programming/svm/liblinear/train"
$liblinear_predict="/home/tsalles/Programming/svm/liblinear/predict"
$libsvm_train="/home/tsalles/Programming/svm/libsvm-3.25/svm-train"
$libsvm_predict="/home/tsalles/Programming/svm/libsvm-3.25/svm-predict"


rm $dataset/preds_bl.out;
rm $dataset/preds_ae.out;
rm $dataset/preds_liblinear_bl.out
rm $dataset/preds_liblinear_ae.out
rm $dataset/preds_libsvm_bl.out
rm $dataset/preds_libsvm_ae.out

for i in `seq 0 $((num_folds-1))`; do
  echo "Round $i"
  python3 model.py -t $dataset/${num_folds}fold/train$i -T $dataset/${num_folds}fold/test$i -o $dataset/preds_bl.out -c 128;
  python3 model.py -t $dataset/${num_folds}fold/train$i -T $dataset/${num_folds}fold/test$i -a 128 64 -b 32  -c 64 --autoencoder --pretrain-ae -E 10 -e 10 -v 0.3 -r $i -p $dataset/hist$i -B $dataset/reduced$i -o $dataset/preds_ae.out -s $dataset/ae_model$i;

  $liblinear_train $dataset/${num_folds}fold/train$i $dataset/svm-model$i 
  $liblinear_predict $dataset/${num_folds}fold/test$i $dataset/svm-model$i $dataset/.svm-preds$i
  utils/svm_predictions.py -i $dataset/.svm-preds$i -t $dataset/${num_folds}fold/test$i -r $i -o $dataset/preds_liblinear_bl.out
  rm $dataset/.svm-preds$i

  $liblinear_train $dataset/reduced${i}_trn $dataset/svm-model$i 
  $liblinear_predict $dataset/reduced${i}_tst $dataset/svm-model$i $dataset/.svm-preds$i
  utils/svm_predictions.py -i $dataset/.svm-preds$i -t $dataset/reduced${i}_tst -r $i -o $dataset/preds_liblinear_ae.out
  rm $dataset/.svm-preds$i 

  $libsvm_train -t 2 $dataset/${num_folds}fold/train$i $dataset/svm-model$i 
  $libsvm_predict $dataset/${num_folds}fold/test$i $dataset/svm-model$i $dataset/.svm-preds$i
  utils/svm_predictions.py -i $dataset/.svm-preds$i -t $dataset/${num_folds}fold/test$i -r $i -o $dataset/preds_libsvm_bl.out
  rm $dataset/.svm-preds$i

  $libsvm_train -t 2 $dataset/reduced${i}_trn $dataset/svm-model$i 
  $libsvm_predict $dataset/reducied${i}_tst $dataset/svm-model$i $dataset/.svm-preds$i
  utils/svm_predictions.py -i $dataset/.svm-preds$i -t $dataset/reducied${i}_tst -r $i -o $dataset/preds_libsvm_ae.out
  rm $dataset/.svm-preds$i
  
done
