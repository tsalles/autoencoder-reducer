# autoencoder-reducer
Experimental code for dimensionality reduction using autoencoders.

## Usage

The best way to understand how to usethe provided script is:
```
python model.py -h
```

Currently, the supported options are described below:
```
usage: model.py [-h] -t TRAIN -T TEST [-v VAL] -o OUTPUT [-s SAVE_MODEL]
                [-p PLOT] [-B BOTTLENECK_OUTPUT] [--autoencoder]
                [--pretrain-ae] [-e CLF_EPOCHS] [-E AE_EPOCHS]
                [-a AE_DIMS [AE_DIMS ...]] [-b BOTTLENECK]
                [-c CLF_DIMS [CLF_DIMS ...]] [-r CV_ROUND] [-l {mse,mae}]

Trains and classifies textual data in TFIDF representation using vanilla NN or
dimensionaly reduced data from AE.

optional arguments:
  -h, --help            show this help message and exit
  -t TRAIN, --train TRAIN
                        Training file in libSVM format.
  -T TEST, --test TEST  Test file in libSVM format.
  -v VAL, --val VAL     Validation file in libSVM format or a validation split
                        on (0,1) open interval.
  -o OUTPUT, --output OUTPUT
                        Output file for predictions.
  -s SAVE_MODEL, --save-model SAVE_MODEL
                        Prefix file for storing trained model.
  -p PLOT, --plot PLOT  Output file preffix for history plot with losses per
                        epoch.
  -B BOTTLENECK_OUTPUT, --bottleneck-output BOTTLENECK_OUTPUT
                        Output file for reduced representation.
  --autoencoder         Uses AE compression before classification, instead of
                        a vanilla NN.
  --pretrain-ae         Pretrains AE then fine tunes the complete model.
  -e CLF_EPOCHS, --clf-epochs CLF_EPOCHS
                        Number of epochs to fit the entire classifier.
  -E AE_EPOCHS, --ae-epochs AE_EPOCHS
                        Number of epochs to fit the AE.
  -a AE_DIMS [AE_DIMS ...], --ae-dims AE_DIMS [AE_DIMS ...]
                        List of AE dimensions.
  -b BOTTLENECK, --bottleneck BOTTLENECK
                        Dimension of bottleneck layer (i.e., the compressed
                        representation).
  -c CLF_DIMS [CLF_DIMS ...], --clf-dims CLF_DIMS [CLF_DIMS ...]
                        List of dense dimensions for classification.
  -r CV_ROUND, --cv-round CV_ROUND
                        Iteration number of cross validation.
  -l {mse,mae}, --loss {mse,mae}
```

Example:

```
python3 model.py -t train_file -T test_file -a 128 64 -b 32  -c 64 -p hist.png \
                 --autoencoder --pretrain-ae -E 10 -e 10 -o predictions.txt -r 0\
                 --loss mae -B reduced_train_file -s model0
```

## Datasets

The datasets can be found on <http://hidra.lbd.dcc.ufmg.br/newdatasets/>.
If you have `xmllint` on your system (for Ubuntu, it can be installed by
`apt install libxml2-utils`), then you can run the provided `utils/download.sh`
script to download all available datasets. The cross validation splits are
compressed as `.gz` files, which can be decompressed by means of the provided
`utils/gunzip.sh` script.

## Experimentation

The `run.sh` script automates the crossed validation procedure, iterating
over all pairs o train/test folds and consolidating the predictions in a single
file with the following format:

```
#0
id_11 real_class_11 predicted_class_11:score_11
id_12 real_class_12 predicted_class_12:score_12
...
id_1x real_class_1x predicted_class_1x:score_1x
#1
id_21 real_class_21 predicted_class_21:score_21
id_22 real_class_22 predicted_class_22:score_22
...
id_2x real_class_2x predicted_class_2x:score_2x

...

#k
id_k1 real_class_k1 predicted_class_k1:score_k1
id_k2 real_class_k2 predicted_class_k2:score_k2
...
id_kx real_class_kx predicted_class_kx:score_kx
```
where x denotes the number of test examples in
the current cross validation fold and k denotes
the number of cross validation folds.

You can also make use of the provided `accuracy3`
script to build the confusion matrix for each
validation round as well as retrieve some metrics
on classification effectiveness (i.e., precision,
recall, microF1 and macroF1). Let predictions.txt
be the output containing the predictions made by
a classifier for k cross validation iterations.
Then, you can compute the confusion matrix as
follows

```
./accuracy3 < predictions.txt
```


