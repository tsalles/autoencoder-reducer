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

def build_model(dim, num_labels, with_ae=True, all_ae_dims=[[256, 128]], bottleneck_dims=[64, 8], clf_dims=[2048, 1024]):
  if with_ae:
    print('all_ae_dims', all_ae_dims)
    print('bottleneck_dims', bottleneck_dims)
    print('clf_dims', clf_dims)
    assert len(all_ae_dims) and sum(len(x) for x in all_ae_dims), 'At least one AE dimension must be specified when using the layered AE'
    assert len(bottleneck_dims), 'At least one bottleneck dimension must be specified when using the layered AE'
    assert len(all_ae_dims) == len(bottleneck_dims), 'The number of AE dimension specs must be equal to the number of bottleneck dimensions'
  assert len(clf_dims), 'At least one CLF dimension must be specified'

  out_layers = []
  bot_layers = []
  input_layer = keras.Input(shape=(dim,))
  if with_ae:
    for ae_idx, ae_dims in enumerate(all_ae_dims):
      bottleneck_dim = bottleneck_dims[ae_idx]
      for i, d in enumerate(ae_dims):
        enc_layer = keras.layers.Dense(d, activation='relu', name='encoder_{}'.format(ae_idx) if i == 0 else None)(input_layer if ae_idx == 0 and i == 0 else bot_layer if  i == 0 else enc_layer)
#        enc_layer = keras.layers.GaussianNoise(0.5)(enc_layer)
        enc_layer = keras.layers.Dropout(0.2)(enc_layer)
        bot_layer = keras.layers.Dense(bottleneck_dim, activation='relu', name='bottleneck_{}'.format(ae_idx))(enc_layer)
      bot_layers.append(bot_layer)
      bot_layer = keras.layers.Concatenate()(bot_layers)
      for i, d in enumerate(reversed(ae_dims)):
        dec_layer = keras.layers.Dense(d, activation='relu')(bot_layer if i == 0 else dec_layer)
      dec_layer = keras.layers.Dense(dim, name='decoder_{}'.format(ae_idx))(dec_layer)
      out_layers.append(dec_layer)
      dim = bottleneck_dim
    bot_layer = keras.layers.Concatenate(name='combined_bottleneck')(bot_layers)

  for i, d in enumerate(clf_dims if with_ae else clf_dims+ae_dims):
    clf_layer = keras.layers.Dense(d, activation='relu')(clf_layer if i > 0 else (bot_layer if with_ae else input_layer))
    clf_layer = keras.layers.Dropout(0.2)(clf_layer)
  
  clf_out_layer = keras.layers.Dense(num_labels, name='classifier', activation='softmax')(clf_layer)

  ae_model = None
  if with_ae:
    ae_model = keras.Model(input_layer, out_layers, name='ae_model')
    ae_model.compile(optimizer='adam', loss=['mae']*len(out_layers))
    compressor = keras.Model(ae_model.input, ae_model.get_layer('combined_bottleneck').output, name='compressor') 

  model = keras.Model(input_layer, out_layers + [clf_out_layer] if with_ae else clf_out_layer, name='ae_clf_model')
  model.compile(optimizer='adam', loss=['mae']*len(out_layers) + ['sparse_categorical_crossentropy'] if with_ae else 'sparse_categorical_crossentropy', loss_weights=[0.2]*len(out_layers) + [0.8] if with_ae else None)
  model.summary()
  keras.utils.plot_model(model, to_file='plot_layered_model.png', show_shapes=True, show_layer_names=True)
  return model, ae_model, compressor


def fit(model, x_trn, y_trn, validation_data=None, clf_epochs=30, ae_epochs=30, with_ae=False, pretrain_ae=False):
  if with_ae and pretrain_ae: # pre-training AE
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=0.01)
    ae_model.fit(x_trn, y_trn, validation_data=validation_data, batch_size=1, epochs=ae_epochs, callbacks=[callback])

  # fine tuning for class separability
  callback = tf.keras.callbacks.EarlyStopping(monitor='classifier_loss' if args.with_ae else 'loss', patience=5, min_delta=0.01)
  model.fit(x_trn, y_trn, validation_data=validation_data, batch_size=1, epochs=clf_epochs, callbacks=[callback])

  return model


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Trains and classifies textual data in TFIDF representation using vanilla NN or dimensionaly reduced data from AE.')
  parser.add_argument('-t', '--train', required=True, help='Training file in libSVM format.')
  parser.add_argument('-T', '--test', required=True, help='Test file in libSVM format.')
  parser.add_argument('-v', '--val', required=False, help='Validation file in libSVM format or a validation split on (0,1) open interval.')
  parser.add_argument('-o', '--output', required=True, help='Output file for predictions.')
  parser.add_argument('--autoencoder', dest='with_ae', action='store_true', help='Uses AE compression before classification, instead of a vanilla NN.')
  parser.add_argument('-B', '--bottleneck-output', required=False, help='Output file for reduced representation.')
  parser.set_defaults(with_ae=False)
  parser.add_argument('--pretrain-ae', dest='pretrain_ae', action='store_true', help='Pretrains AE then fine tunes the complete model')
  parser.add_argument('-e', '--clf-epochs', type=int, help='Number of epochs for classifier training', default=20)
  parser.add_argument('-E', '--ae-epochs', type=int, help='Number of epochs for AE training', default=20)
  parser.add_argument('-a', '--ae-dims', nargs='+', type=int, action='append', help='List of each layered AE dimensions')
  parser.add_argument('-b', '--bottleneck', nargs='+', type=int, help='Dimensions of each bottleneck layer (i.e., the compressed representation)')
  parser.add_argument('-c', '--clf-dims', nargs='+', type=int, help='List of dense dimensions for classification')
  parser.set_defaults(with_ae=False)
  parser.set_defaults(pretrain_ae=False)

  args = parser.parse_args()

  labels, x_trn, y_trn, x_tst, y_tst, x_val, y_val = parse(args.train, args.test, args.val)
  validation_data = (x_val, y_val) if  args.val and x_trn.shape[0] > 0 else None

  model, ae_model, compressor = build_model(x_trn.shape[1], len(labels), with_ae=args.with_ae, all_ae_dims=args.ae_dims, bottleneck_dims=args.bottleneck, clf_dims=args.clf_dims)
  model = fit(model, x_trn, y_trn, validation_data=validation_data, clf_epochs=args.clf_epochs, ae_epochs=args.ae_epochs, with_ae=args.with_ae, pretrain_ae=args.pretrain_ae)

  if not args.bottleneck_output:
    if args.with_ae:
      if args.pretrain_ae:
        ae_model.trainable = True
        callback = tf.keras.callbacks.EarlyStopping(monitor='classifier_loss' if args.with_ae else 'loss', patience=5, min_delta=0.01)
        model.fit(x_trn, y_trn, validation_data, batch_size=1, epochs=args.clf_epochs, callbacks=[callback])
      y_prd = model.predict(x_tst, batch_size=1)[-1]
    else:
      y_prd = model.predict(x_tst, batch_size=1)
  
    with io.open(args.output, 'a') as fh:
      fh.write('#{}\n'.format(args.train.replace('train', '')))
      for i, (y_t, y_p) in enumerate(zip(y_tst, y_prd)):
        fh.write('{} {} {}:{}\n'.format(i, y_t, np.argmax(y_p), np.max(y_p)))
  elif compressor is not None:
    # FIXME MOVE SHUFFLINH FROM PARSER TO THIS FILE!!!
    x_new = compressor.predict(x_trn if not validation_data else sp.vstack((x_trn, x_val), format='csr'))
    with io.open(args.bottleneck_output, 'w') as fh:
      for i, x in enumerate(x_new):
        fh.write('{} {}\n'.format(y_trn[i], ' '.join('{}:{}'.format(j, v) for j, v in enumerate(x))))

