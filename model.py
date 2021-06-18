#fixed_seed=42
#from numpy.random import seed
#seed(fixed_seed)
#from tensorflow import set_random_seed
#set_random_seed(fixed_seed)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse
from parse import parse
import io
import scipy as sp


def build_model(dim, num_labels, with_ae=True, ae_dims=[256, 128], bottleneck_dim=64, clf_dims=[2048, 1024]):
  if with_ae:
    assert len(ae_dims), 'At least one AE dimension must be specified when using the AE'
  assert len(clf_dims), 'At least one CLF dimension must be specified'

  ae_model, compressor = None, None
  input_layer = keras.Input(shape=(dim,))
  
  if with_ae:
    for i, d in enumerate(ae_dims):
      enc_layer = keras.layers.Dense(d, activation='relu')(input_layer if i == 0 else enc_layer)
#      enc_layer = keras.layers.GaussianNoise(0.2)(enc_layer)
      #enc_layer = keras.layers.Dropout(0.5)(enc_layer)
    bot_layer = keras.layers.Dense(bottleneck_dim, activation='relu', name='bottleneck')(enc_layer)
    for i, d in enumerate(reversed(ae_dims)):
      dec_layer = keras.layers.Dense(d, activation='relu')(bot_layer if i == 0 else dec_layer)
    dec_layer = keras.layers.Dense(dim, name='decoder')(dec_layer)

  for i, d in enumerate(clf_dims if with_ae else clf_dims+ae_dims):
    clf_layer = keras.layers.Dense(d, activation='relu')(clf_layer if i > 0 else (bot_layer if with_ae else input_layer))
    #clf_layer = keras.layers.Dropout(0.5)(clf_layer)
  clf_out_layer = keras.layers.Dense(num_labels, name='classifier', activation='softmax')(clf_layer)

  ae_model = None
  if with_ae:
    ae_model = keras.Model(input_layer, dec_layer, name='ae_model')
    ae_model.compile(optimizer='adam', loss='mae')
    compressor = keras.Model(ae_model.input, ae_model.get_layer('bottleneck').output, name='compressor')


  model = keras.Model(input_layer, [dec_layer, clf_out_layer] if with_ae else clf_out_layer, name='ae_clf_model')
  model.compile(optimizer='adam', loss=['mae', 'sparse_categorical_crossentropy'], loss_weights=[0.4, 0.6])
  model.summary()

  return model, ae_model, compressor


def fit(model, x_trn, y_trn, validation_data=None, clf_epochs=30, ae_epochs=30, with_ae=False, pretrain_ae=False):
  if with_ae and pretrain_ae: # pre-training AE
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=0.01)
    ae_model.fit(x_trn, y_trn, validation_data=validation_data, batch_size=1, epochs=ae_epochs, callbacks=[callback])

  # fine tuning for class separability
  callback = tf.keras.callbacks.EarlyStopping(monitor='classifier_loss' if args.with_ae else 'loss', patience=5, min_delta=0.01)
  h = model.fit(x_trn, y_trn, validation_data=validation_data, batch_size=1, epochs=clf_epochs, callbacks=[callback])

  return model, h


def plot(model, history, history_fn):
  import matplotlib.pyplot as plt

  keras.utils.plot_model(model, to_file='arch_{}'.format(history_fn), show_shapes=True, show_layer_names=True)

  fig, (ax1, ax2) = plt.subplots(2)
  fig.suptitle('Training history')

  ax1.plot(history.history['accuracy'])
  ax1.plot(history.history['val_accuracy'])
  ax1.title('Model Accuracy')
  ax1.ylabel('accuracy')
  ax1.xlabel('epoch')
  ax1.legend(['trn', 'val'], loc='upper left')

  ax2.plot(history.history['loss'])
  ax2.plot(history.history['val_loss'])
  ax2.title('Model Loss')
  ax2.ylabel('loss')
  ax2.xlabel('epoch')
  ax2.legend(['trn', 'tst'], loc='upper left')

  plt.savefig(history_fn)


# TODO: output bootleneck layer representation to a file!!!
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Trains and classifies textual data in TFIDF representation using vanilla NN or dimensionaly reduced data from AE.')
  parser.add_argument('-t', '--train', required=True, help='Training file in libSVM format.')
  parser.add_argument('-T', '--test', required=True, help='Test file in libSVM format.')
  parser.add_argument('-v', '--val', required=False, help='Validation file in libSVM format or a validation split on (0,1) open interval.')
  parser.add_argument('-o', '--output', required=True, help='Output file for predictions.')
  parser.add_argument('-p', '--plot', required=False, help='Output file for history plot with losses per epoch.')
  parser.add_argument('-B', '--bottleneck-output', required=False, help='Output file for reduced representation.')
  parser.add_argument('--autoencoder', dest='with_ae', action='store_true', help='Uses AE compression before classification, instead of a vanilla NN.')
  parser.set_defaults(with_ae=False)
  parser.add_argument('--pretrain-ae', dest='pretrain_ae', action='store_true', help='Pretrains AE then fine tunes the complete model.')
  parser.add_argument('-e', '--clf-epochs', type=int, help='Number of epochs to fit the entire classifier.', default=20)
  parser.add_argument('-E', '--ae-epochs', type=int, help='Number of epochs to fit the AE.', default=20)
  parser.add_argument('-a', '--ae-dims', nargs='+', type=int, help='List of AE dimensions.', default=[256, 128])
  parser.add_argument('-b', '--bottleneck', type=int, help='Dimension of bottleneck layer (i.e., the compressed representation).', default=64)
  parser.add_argument('-c', '--clf-dims', nargs='+', type=int, help='List of dense dimensions for classification.', default=[4096, 2048])
  parser.add_argument('-r', '--cv-round', type=int, help='Iteration number of cross validation.', default=0)
  parser.set_defaults(with_ae=False)
  parser.set_defaults(pretrain_ae=False)

  args = parser.parse_args()

  le, x_trn, y_trn, x_tst, y_tst, x_val, y_val = parse(args.train, args.test, args.val)
  validation_data = (x_val, y_val) if  args.val and x_trn.shape[0] > 0 else None

  model, ae_model, compressor = build_model(x_trn.shape[1], len(le.classes_), with_ae=args.with_ae, ae_dims=args.ae_dims, bottleneck_dim=args.bottleneck, clf_dims=args.clf_dims)
  model, h = fit(model, x_trn, y_trn, validation_data=validation_data, clf_epochs=args.clf_epochs, ae_epochs=args.ae_epochs, with_ae=args.with_ae, pretrain_ae=args.pretrain_ae)


  if validation_data:
    model, _ = fit(model, validation_data[0], validation_data[1], clf_epochs=args.clf_epochs, ae_epochs=args.ae_epochs, with_ae=args.with_ae, pretrain_ae=args.pretrain_ae)

  
  if args.with_ae:
    _, y_prd = model.predict(x_tst)
  else:
    y_prd = model.predict(x_tst)
  
  with io.open(args.output, 'a') as fh:
    preds = le.inverse_transform(np.argmax(y_prd, axis=1))
    scores = np.max(y_prd, axis=1)
    print([(i, c) for i, c in enumerate(le.classes_)])
    fh.write('#{}\n'.format(args.cv_round))
    for i, y_t in enumerate(y_tst):
      fh.write('{} {} {}:{}\n'.format(i, y_t, preds[i], scores[i]))
  if args.bottleneck_output and compressor is not None:
    x_new = compressor.predict(x_trn if not validation_data else sp.vstack((x_trn, x_val), format='csr'))
    with io.open(args.bottleneck_output, 'w') as fh:
      for i, x in enumerate(x_new):
        fh.write('{} {}\n'.format(y_trn[i], ' '.join('{}:{}'.format(j, v) for j, v in enumerate(x))))
        
  if args.plot:
    fn = args.plot if args.plot.endswith('.png') else args.plot.split('.')[0]+'.png'
    plot(model, h, args.plot)

