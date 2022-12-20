#!/bin/bash

export TF_GPU_ALLOCATOR=cuda_malloc_async

dataset=$1
num_folds=$2

liblinear_train="/home/tsalles/Programming/svm/liblinear/train"
liblinear_predict="/home/tsalles/Programming/svm/liblinear/predict"
libsvm_train="/home/tsalles/Programming/svm/libsvm-3.25/svm-train"
libsvm_predict="/home/tsalles/Programming/svm/libsvm-3.25/svm-predict"

outdir=$dataset/run;
mkdir -p $outdir;

rm $outdir/*;

for i in `seq 0 $((num_folds-1))`; do
  echo "Round $i"

  python3 preprocessing/parse.py -p $dataset -F $i -f tecbench -o $dataset/fold_${i}/libsvm;

  c=`$liblinear_train -C -s 2 $dataset/fold_${i}/libsvm_trn.svm | tail -n 1 | awk '{print $4}'`;
  $liblinear_train -c $c -s 2 $dataset/fold_${i}/libsvm_trn.svm $outdir/svm-model$i
  $liblinear_predict $dataset/fold_${i}/libsvm_tst.svm $outdir/svm-model$i $outdir/.svm-preds$i
  python3 utils/svm_predictions.py -i $outdir/.svm-preds$i -t $dataset/fold_${i}/libsvm_tst.svm -r $i -o $outdir/preds_liblinear_original-full.out
  rm $outdir/.svm-preds$i

  python3 shallow_classify.py -t $dataset/fold_${i}/libsvm_trn.svm -T $dataset/fold_${i}/libsvm_tst.svm -r $i -m xb -o $outdir/preds_xb_original-full.out
  python3 shallow_classify.py -t $dataset/fold_${i}/libsvm_trn.svm -T $dataset/fold_${i}/libsvm_tst.svm -r $i -m rf -o $outdir/preds_rf_original-full.out
  python3 shallow_classify.py -t $dataset/fold_${i}/libsvm_trn.svm -T $dataset/fold_${i}/libsvm_tst.svm -r $i -m xt -o $outdir/preds_xt_original-full.out

  for m in `seq 5 2 11`; do
    c=`echo "x=(-3 + sqrt(9+16*(2^$m)))/8; scale=0; x/1+1" | bc -l`;
    b=`echo "2*$c" | bc`;
    a=`echo "2^$m - $b - $c" | bc`;
    dims=`echo "$a+$b+$c" | bc`;

    echo "Evaluating results with $a -> $b -> $c = $dims dimensions [i=$m]";

    python3 reduce_dim.py -P $dataset/ -r 0 -d $dims -m svd -o $outdir/reduced-svd-$dims-${i};
    python3 reduce_dim.py -P $dataset/ -r 0 -d $dims -m nmf -o $outdir/reduced-nmf-$dims-${i};
    python3 reduce_dim.py -P $dataset/ -r 0 -d $dims -m chi -o $outdir/reduced-chi-$dims-${i};
    python3 reduce_dim.py -P $dataset/ -r 0 -d $dims -m mic -o $outdir/reduced-mic-$dims-${i};

    python3 hae.py -P $dataset -E 50 -a $(( a * 4 )) $(( a * 2 )) $a -b $dims -r $i -f tecbench -v 0.3 -o $outdir/preds_ae_hae-$dims.out  -B $outdir/reduced-hae-$dims-$i

    for variant in reduced-hae reduced-svd reduced-nmf reduced-chi reduced-mic; do  # reduced-lae
      c=`$liblinear_train -C -s 2 $outdir/$variant-$dims-${i}_trn | tail -n 1 | awk '{print $4}'`;
      $liblinear_train -c $c -s 2 $outdir/$variant-$dims-${i}_trn $outdir/svm-model$i
      $liblinear_predict $outdir/$variant-$dims-${i}_tst $outdir/svm-model$i $outdir/.svm-preds$i
      python3 utils/svm_predictions.py -i $outdir/.svm-preds$i -t $outdir/$variant-$dims-${i}_tst -r $i -o $outdir/preds_liblinear_$variant-$dims.out
      rm $outdir/.svm-preds$i

      python3 shallow_classify.py -t $outdir/$variant-$dims-${i}_trn -T $outdir/$variant-$dims-${i}_tst -r $i -m xb -o $outdir/preds_xb_$variant-$dims.out
      python3 shallow_classify.py -t $outdir/$variant-$dims-${i}_trn -T $outdir/$variant-$dims-${i}_tst -r $i -m rf -o $outdir/preds_rf_$variant-$dims.out
      python3 shallow_classify.py -t $outdir/$variant-$dims-${i}_trn -T $outdir/$variant-$dims-${i}_tst -r $i -m xt -o $outdir/preds_xt_$variant-$dims.out
    done

  done

done

exit;

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
  python3 model_layered_clf.py -P $dataset/ -f tecbench -a 1024 512 256 128 -b 32 -c 512 --autoencoder --pretrain-ae -E 50 -e 50 -v 0.3 -r $i -p $dataset/hist_ae$i -B $dataset/reduced-lae$i -o $dataset/preds_ae.out -s $dataset/ae_model$i -S 64 -B $dataset/reduced-lae$i;

  python3 model_layered_clf.py -P $dataset/ -f tecbench -a 1024 512 -a 256 128 -a 64 32 -b 128 64 32 -c 512 --autoencoder --pretrain-ae -E 50 -e 50 -v 0.3 -r $i -p $dataset/hist_layered$i -B $dataset/reduced-hae$i -o $dataset/preds_ae_layered.out -s $dataset/ae_model_layered$i -S 64 -B $dataset/reduced-hae$i;

  python3 preprocessing/parse.py -p $dataset -F $i -f tecbench -o $dataset/fold_${i}/libsvm;

  c=`$liblinear_train -C -s 2 $dataset/fold_${i}/libsvm_trn.svm | tail -n 1 | awk '{print $4}'`;
  $liblinear_train -c $c -s 2 $dataset/fold_${i}/libsvm_trn.svm $dataset/svm-model$i
  $liblinear_predict $dataset/fold_${i}/libsvm_tst.svm $dataset/svm-model$i $dataset/.svm-preds$i
  python3 utils/svm_predictions.py -i $dataset/.svm-preds$i -t $dataset/fold_${i}/libsvm_tst.svm -r $i -o $dataset/preds_liblinear_bl.out
  rm $dataset/.svm-preds$i

  c=`$liblinear_train -C -s 2 $dataset/reduced-lae${i}_trn | tail -n 1 | awk '{print $4}'`;
  $liblinear_train -c $c -s 2 $dataset/reduced-lae${i}_trn $dataset/svm-model$i
  $liblinear_predict $dataset/reduced-lae${i}_tst $dataset/svm-model$i $dataset/.svm-preds$i
  python3 utils/svm_predictions.py -i $dataset/.svm-preds$i -t $dataset/reduced-lae${i}_tst -r $i -o $dataset/preds_liblinear_ae_lae.out
  rm $dataset/.svm-preds$i

  c=`$liblinear_train -C -s 2 $dataset/reduced-hae${i}_trn | tail -n 1 | awk '{print $4}'`;
  $liblinear_train -c $c -s 2 $dataset/reduced-hae${i}_trn $dataset/svm-model$i
  $liblinear_predict $dataset/reduced-hae${i}_tst $dataset/svm-model$i $dataset/.svm-preds$i
  python3 utils/svm_predictions.py -i $dataset/.svm-preds$i -t $dataset/reduced-hae${i}_tst -r $i -o $dataset/preds_liblinear_ae_hae.out
  rm $dataset/.svm-preds$i

  #$libsvm_train -t 2 $dataset/fold_${i}/libsvm_trn.svm $dataset/svm-model$i
  #$libsvm_predict $dataset/fold_${i}/libsvm_tst.svm $dataset/svm-model$i $dataset/.svm-preds$i
  #python3 utils/svm_predictions.py -i $dataset/.svm-preds$i -t $dataset/fold_${i}/libsvm_tst.svm -r $i -o $dataset/preds_libsvm_bl.out
  #rm $dataset/.svm-preds$i

  #$libsvm_train -t 2 $dataset/reduced-hae${i}_trn $dataset/svm-model$i
  #$libsvm_predict $dataset/reduced-hae${i}_tst $dataset/svm-model$i $dataset/.svm-preds$i
  #python3 utils/svm_predictions.py -i $dataset/.svm-preds$i -t $dataset/reduced-hae${i}_tst -r $i -o $dataset/preds_libsvm_ae_layered.out
  #rm $dataset/.svm-preds$i

done
