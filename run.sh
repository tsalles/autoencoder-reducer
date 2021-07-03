#!/bin/bash

dataset=$1
num_folds=$2

liblinear_train="/Users/tsalles/Programming/autoencoder-reducer/liblinear/train"
liblinear_predict="/Users/tsalles/Programming/autoencoder-reducer/liblinear/predict"
libsvm_train="/Users/tsalles/Programming/autoencoder-reducer/libsvm/svm-train"
libsvm_predict="/Users/tsalles/Programming/autoencoder-reducer/libsvm/svm-predict"


rm $dataset/preds_bl.out;
rm $dataset/preds_ae.out;
rm $dataset/preds_liblinear_bl.out
rm $dataset/preds_liblinear_ae.out
rm $dataset/preds_libsvm_bl.out
rm $dataset/preds_libsvm_ae.out

for i in `seq 0 $((num_folds-1))`; do
  echo "Round $i"
  python3 model.py -t $dataset/${num_folds}fold/train$i -T $dataset/${num_folds}fold/test$i -o $dataset/preds_bl.out -c 256 -S 64 -e 50 -v 0.3 -r $i -p $dataset/hist_bl$i;
  python3 model.py -t $dataset/${num_folds}fold/train$i -T $dataset/${num_folds}fold/test$i -a 512 256 -b 128  -c 256 --autoencoder --pretrain-ae -E 50 -e 50 -v 0.3 -r $i -p $dataset/hist$i -B $dataset/reduced$i -o $dataset/preds_ae.out -s $dataset/ae_model$i -S 64;
  python3 model_layered.py -t $dataset/${num_folds}fold/train$i -T $dataset/${num_folds}fold/test$i -a 512 256 -a 128 64 -a 32 16 -b 64 32 16 -c 256  --autoencoder --pretrain-ae -E 50 -e 50 -v 0.3 -r $i -p $dataset/hist_layered$i -B $dataset/reduced_layered$i -o $dataset/preds_ae_layered.out -s $dataset/ae_model_layered$i -S 64 -B $dataset/reduced_layered$i;


  $liblinear_train $dataset/${num_folds}fold/train$i $dataset/svm-model$i 
  $liblinear_predict $dataset/${num_folds}fold/test$i $dataset/svm-model$i $dataset/.svm-preds$i
  python3 utils/svm_predictions.py -i $dataset/.svm-preds$i -t $dataset/${num_folds}fold/test$i -r $i -o $dataset/preds_liblinear_bl.out
  rm $dataset/.svm-preds$i

  $liblinear_train $dataset/reduced${i}_trn $dataset/svm-model$i 
  $liblinear_predict $dataset/reduced${i}_tst $dataset/svm-model$i $dataset/.svm-preds$i
  python3 utils/svm_predictions.py -i $dataset/.svm-preds$i -t $dataset/reduced${i}_tst -r $i -o $dataset/preds_liblinear_ae.out
  rm $dataset/.svm-preds$i 

  $liblinear_train $dataset/reduced_layered${i}_trn $dataset/svm-model$i 
  $liblinear_predict $dataset/reduced_layered${i}_tst $dataset/svm-model$i $dataset/.svm-preds$i
  python3 utils/svm_predictions.py -i $dataset/.svm-preds$i -t $dataset/reduced_layered${i}_tst -r $i -o $dataset/preds_liblinear_ae_layered.out
  rm $dataset/.svm-preds$i 


  $libsvm_train -t 2 $dataset/${num_folds}fold/train$i $dataset/svm-model$i 
  $libsvm_predict $dataset/${num_folds}fold/test$i $dataset/svm-model$i $dataset/.svm-preds$i
  python3 utils/svm_predictions.py -i $dataset/.svm-preds$i -t $dataset/${num_folds}fold/test$i -r $i -o $dataset/preds_libsvm_bl.out
  rm $dataset/.svm-preds$i

  $libsvm_train -t 2 $dataset/reduced_layered${i}_trn $dataset/svm-model$i 
  $libsvm_predict $dataset/reduced_layered${i}_tst $dataset/svm-model$i $dataset/.svm-preds$i
  python3 utils/svm_predictions.py -i $dataset/.svm-preds$i -t $dataset/reduced_layered${i}_tst -r $i -o $dataset/preds_libsvm_ae_layered.out
  rm $dataset/.svm-preds$i
  
done
