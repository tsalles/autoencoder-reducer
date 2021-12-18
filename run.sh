#!/bin/bash

dataset=$1
num_folds=$2

liblinear_train="/home/tsalles/Programming/svm/liblinear/train"
liblinear_predict="/home/tsalles/Programming/svm/liblinear/predict"
libsvm_train="/home/tsalles/Programming/svm/libsvm-3.25/svm-train"
libsvm_predict="/home/tsalles/Programming/svm/libsvm-3.25/svm-predict"


rm $dataset/preds_bl.out;
rm $dataset/preds_ae.out;
rm $dataset/preds_ae_layered.out;
rm $dataset/preds_liblinear_bl.out
rm $dataset/preds_liblinear_ae.out
rm $dataset/preds_liblinear_ae_layered.out
rm $dataset/preds_libsvm_bl.out
rm $dataset/preds_libsvm_ae.out
rm $dataset/preds_libsvm_ae_layered.out

for i in `seq 0 $((num_folds-1))`; do
  echo "Round $i"
##  python3 model.py -t $dataset/fold_${i}/train$i -T $dataset/fold_${i}/test$i -o $dataset/preds_bl.out -c 512 -S 64 -e 50 -v 0.3 -r $i -p $dataset/hist_bl$i;
##  python3 model.py -t $dataset/fold_${i}/train$i -T $dataset/fold_${i}/test$i -a 1024 512 -b 256  -c 512 --autoencoder --pretrain-ae -E 50 -e 50 -v 0.3 -r $i -p $dataset/hist$i -B $dataset/reduced$i -o $dataset/preds_ae.out -s $dataset/ae_model$i -S 64;

  python3 model_layered_clf.py -P $dataset/ -f tecbench -a 1024 512 -a 256 128 -a 64 32 -b 128 64 32 -c 512 --autoencoder --pretrain-ae -E 50 -e 50 -v 0.3 -r $i -p $dataset/hist_layered$i -B $dataset/reduced_layered$i -o $dataset/preds_ae_layered.out -s $dataset/ae_model_layered$i -S 64 -B $dataset/reduced_layered$i;

  python3 preprocessing/parse.py -p $dataset -F $i -f tecbench -o $dataset/fold_${i}/libsvm;

  $liblinear_train $dataset/fold_${i}/libsvm_trn.svm $dataset/svm-model$i
  $liblinear_predict $dataset/fold_${i}/libsvm_tst.svm $dataset/svm-model$i $dataset/.svm-preds$i
  python3 utils/svm_predictions.py -i $dataset/.svm-preds$i -t $dataset/fold_${i}/libsvm_tst.svm -r $i -o $dataset/preds_liblinear_bl.out
  rm $dataset/.svm-preds$i

  $liblinear_train $dataset/reduced${i}_trn $dataset/svm-model$i
  $liblinear_predict $dataset/reduced${i}_tst $dataset/svm-model$i $dataset/.svm-preds$i
  python3 utils/svm_predictions.py -i $dataset/.svm-preds$i -t $dataset/reduced${i}_tst -r $i -o $dataset/preds_liblinear_ae.out
  rm $dataset/.svm-preds$i

  $liblinear_train $dataset/reduced_layered${i}_trn $dataset/svm-model$i
  $liblinear_predict $dataset/reduced_layered${i}_tst $dataset/svm-model$i $dataset/.svm-preds$i
  python3 utils/svm_predictions.py -i $dataset/.svm-preds$i -t $dataset/reduced_layered${i}_tst -r $i -o $dataset/preds_liblinear_ae_layered.out
  rm $dataset/.svm-preds$i


  $libsvm_train -t 2 $dataset/fold_${i}/libsvm_trn.svm $dataset/svm-model$i
  $libsvm_predict $dataset/fold_${i}/libsvm_tst.svm $dataset/svm-model$i $dataset/.svm-preds$i
  python3 utils/svm_predictions.py -i $dataset/.svm-preds$i -t $dataset/fold_${i}/libsvm_tst.svm -r $i -o $dataset/preds_libsvm_bl.out
  rm $dataset/.svm-preds$i

  $libsvm_train -t 2 $dataset/reduced_layered${i}_trn $dataset/svm-model$i
  $libsvm_predict $dataset/reduced_layered${i}_tst $dataset/svm-model$i $dataset/.svm-preds$i
  python3 utils/svm_predictions.py -i $dataset/.svm-preds$i -t $dataset/reduced_layered${i}_tst -r $i -o $dataset/preds_libsvm_ae_layered.out
  rm $dataset/.svm-preds$i

done
