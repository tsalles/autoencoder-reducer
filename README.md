# autoencoder-reducer
Experimental code for dimensionality reduction using autoencoders.

## Usage

```
python model.py -h
```

```
usage: model.py [-h] -t TRAIN -T TEST [-v VAL] -o OUTPUT
                [-B BOTTLENECK_OUTPUT] [--autoencoder] [--pretrain-ae]
                [-e CLF_EPOCHS] [-E AE_EPOCHS] [-a AE_DIMS [AE_DIMS ...]]
                [-b BOTTLENECK] [-c CLF_DIMS [CLF_DIMS ...]]

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
  -B BOTTLENECK_OUTPUT, --bottleneck-output BOTTLENECK_OUTPUT
                        Output file for reduced representation.
  --autoencoder         Uses AE compression before classification, instead of
                        a vanilla NN.
  --pretrain-ae         Pretrains AE then fine tunes the complete model
  -e CLF_EPOCHS, --clf-epochs CLF_EPOCHS
                        Number of epochs to fit the entire classifier
  -E AE_EPOCHS, --ae-epochs AE_EPOCHS
                        Number of epochs to fit the AE
  -a AE_DIMS [AE_DIMS ...], --ae-dims AE_DIMS [AE_DIMS ...]
                        List of AE dimensions
  -b BOTTLENECK, --bottleneck BOTTLENECK
                        Dimension of bottleneck layer (i.e., the compressed
                        representation)
  -c CLF_DIMS [CLF_DIMS ...], --clf-dims CLF_DIMS [CLF_DIMS ...]
                        List of dense dimensions for classification
```
