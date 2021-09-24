#fixed_seed=42
#from numpy.random import seed
#seed(fixed_seed)
#from tensorflow import set_random_seed
#set_random_seed(fixed_seed)

import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse

from preprocessing.parse import parse
from preprocessing.datamodule.TecDataModule import TeCDataModule
from preprocessing.helper.EvalHelper import EvalHelper

import io
import scipy as sp
import pickle


def l1l2(l1_weight=1., l2_weight=1.):
    l1_weight = l1_weight / (l1_weight+l2_weight)
    l2_weight = l2_weight / (l1_weight+l2_weight)
    def loss(y_true,y_pred):
      mse = keras.losses.mean_squared_error(y_true, y_pred)
      mae = keras.losses.mean_absolute_error(y_true, y_pred)
      return l1_weight*mae + l2_weight*mae
    return loss


def build_model(dim, num_labels, with_ae=True, all_ae_dims=[[256, 128]], bottleneck_dims=[64, 8], clf_dims=[2048, 1024], loss='mse'):
  if with_ae:
    #assert len(all_ae_dims) and sum(len(x) for x in all_ae_dims), 'At least one AE dimension must be specified when using the layered AE'
    assert len(bottleneck_dims), 'At least one bottleneck dimension must be specified when using the layered AE'
    assert len(all_ae_dims) == len(bottleneck_dims), 'The number of AE dimension specs must be equal to the number of bottleneck dimensions'
  assert with_ae or len(clf_dims), 'At least one CLF dimension must be specified when not using AE architecture'

  out_layers = []
  clf_layers = []
  input_layer = keras.Input(shape=(dim,))
  if with_ae:
    enc_layer, bot_layer, dec_layer = None, None, None
    bot_layers = []
    for ae_idx, ae_dims in enumerate(all_ae_dims):
      #assert len(ae_dims), 'Dimensions for AE {} must be specified'.format(ae_idx)
      bottleneck_dim = bottleneck_dims[ae_idx]
      for i, d in enumerate(ae_dims):
        prev_layer = input_layer if ae_idx == 0 and i == 0 else bot_layer if  i == 0 else enc_layer
        enc_layer = keras.layers.Dense(d, activation='relu', name='encoder_{}'.format(ae_idx) if i == 0 else None)(prev_layer)
#        enc_layer = keras.layers.GaussianNoise(0.5)(enc_layer)
#        enc_layer = keras.layers.Dropout(0.3)(enc_layer)
      if not ae_dims:
        enc_layer = bot_layers[-1] if bot_layers else input_layer
      bot_layer = keras.layers.Dense(bottleneck_dim, activation='relu', name='bottleneck_{}'.format(ae_idx))(enc_layer)
      bot_layers.append(bot_layer)
      for i, d in enumerate(reversed(ae_dims)):
        dec_layer = keras.layers.Dense(d, activation='relu')(bot_layer if i == 0 else dec_layer)
      dec_layer = keras.layers.Dense(dim, name='decoder_{}'.format(ae_idx))(bot_layer if not ae_dims else dec_layer)

      out_layers.append(dec_layer)
      dim = bottleneck_dim

#      for i, d in enumerate(clf_dims) if clf_dims else []:
#        clf_layer = keras.layers.Dense(d, activation='relu')(dec_layer if i == 0 else clf_layer)
##        clf_layer = keras.layers.Dropout(0.3)(clf_layer)
      clf_layer = keras.layers.Dense(num_labels, name='loc_classifier_{}'.format(ae_idx), activation='softmax')(dec_layer) #clf_layer if clf_dims else dec_layer)
      clf_layers.append(clf_layer)

    bot_layer = keras.layers.Concatenate(name='combined_bottleneck', axis=1)(bot_layers)

#  for i, d in enumerate(clf_dims) if clf_dims else []:
#    clf_layer = keras.layers.Dense(d, activation='relu')(input_layer if not with_ae else (clf_layer if i > 0 else (bot_layer if with_ae else input_layer)))
##    clf_layer = keras.layers.Dropout(0.3)(clf_layer)
  clf_out_layer = keras.layers.Dense(num_labels, name='glb_classifier', activation='softmax')(bot_layer if with_ae else input_layer) #(clf_layer if clf_dims else (bot_layer if with_ae else input_layer))
  clf_layers.append(clf_out_layer)

  model = keras.Model(input_layer, out_layers + clf_layers if with_ae else clf_layers, name='ae_clf_model')
  model.compile(optimizer='adam', loss=[loss]*len(out_layers) + ['sparse_categorical_crossentropy']*len(clf_layers) if with_ae else 'sparse_categorical_crossentropy',
#                metrics=[['mse', 'mae']]*len(out_layers) + ['accuracy', 'sparse_top_k_categorical_accuracy'] if with_ae else ['accuracy', 'sparse_top_k_categorical_accuracy'],
                loss_weights=[0.5]*len(out_layers) + [0.5]*len(clf_layers[:-1])+[0.5] if with_ae else None)
  model.summary()
  keras.utils.plot_model(model, to_file='plot_layered_model.png', show_shapes=True, show_layer_names=True)

  ae_model, compressor = None, None
  if with_ae:
    ae_model = keras.Model(input_layer, out_layers + clf_layers[:-1], name='ae_model')
    ae_model.compile(optimizer='adam', loss=[loss]*len(out_layers) + ['sparse_categorical_crossentropy']*(len(clf_layers)-1),
                     loss_weights=[0.5]*len(out_layers) + [0.5]*(len(clf_layers)-1))
    ae_model.summary()
    keras.utils.plot_model(ae_model, to_file='plot_layered_ae.png', show_shapes=True, show_layer_names=True)
    compressor = keras.Model(ae_model.input, model.get_layer('combined_bottleneck').output, name='compressor')
  return model, ae_model, compressor


def fit(model, x_trn, y_trn, validation_data=None, clf_epochs=30, ae_epochs=30, with_ae=False, pretrain_ae=False, batch_size=16, use_tb_callback=False):
  if with_ae: # pre-training AE
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss' if validation_data else 'loss', patience=5, min_delta=0.01)]
    if use_tb_callback:
      callbacks.append(tf.keras.callbacks.TensorBoard(log_dir="./logs-trn"))
    ae_model.fit(x_trn, y_trn, validation_data=validation_data, batch_size=batch_size, epochs=ae_epochs, callbacks=callbacks)
    if pretrain_ae:
      ae_model.trainable = False

  # fine tuning for class separability
  callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_glb_classifier_loss' if validation_data and args.with_ae else ('glb_classifier_loss' if args.with_ae else ('val_loss' if validation_data else 'loss')), patience=5, min_delta=0.01)]
  if use_tb_callback:
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir="./logs-trn-ft"))
  h = model.fit(x_trn, y_trn, validation_data=validation_data, batch_size=batch_size, epochs=clf_epochs, callbacks=callbacks)

  return model, h


def plot(model, history, history_fn):
  import matplotlib.pyplot as plt

  keras.utils.plot_model(model, to_file='{}_arch.png'.format(history_fn), show_shapes=True, show_layer_names=True)

  fig, axs = plt.subplots(2, 2)
  fig.suptitle('Training history')

  for metric, ax, model in [('accuracy', axs[0, 0], 'classifier'), ('loss', axs[0, 1], 'classifier'),
                            ('mse', axs[1, 0], 'decoder'), ('loss', axs[1, 1], 'decoder')]:
    lgd = []
    if '{}_{}'.format(model, metric) in history.history:
      ax.plot(history.history['{}_{}'.format(model, metric)])
      lgd.append('trn')
    elif metric in history.history:
      ax.plot(history.history[metric])
      lgd.append('trn')
    if 'val_{}_{}'.format(model, metric) in history.history:
      ax.plot(history.history['val_{}_{}'.format(model, metric)])
      lgd.append('val')
    elif 'val_{}'.format(metric) in history.history:
      ax.plot(history.history['val_{}'.format(metric)])
      lgd.append('val')

    ax.set_title('Model {}'.format(metric))
    ax.set_ylabel(metric)
    ax.set_xlabel('epoch')
    ax.legend(['trn', 'val'], loc='upper left')

  plt.savefig('{}_convergence.png'.format(history_fn))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Trains and classifies textual data in TFIDF representation using vanilla NN or dimensionaly reduced data from AE.')
  parser.add_argument('-t', '--train', help='Training file in libSVM format.')
  parser.add_argument('-T', '--test', help='Test file in libSVM format.')
  parser.add_argument('-v', '--val', help='Validation file in libSVM format or a validation split on (0,1) open interval.')
  parser.add_argument('-o', '--output', required=True, help='Output file for predictions.')
  parser.add_argument('-s', '--save-model', required=False, help='Prefix file for storing trained model.')
  parser.add_argument('-p', '--plot', required=False, help='Output file preffix for history plot with losses per epoch.')
  parser.add_argument('--autoencoder', dest='with_ae', action='store_true', help='Uses AE compression before classification, instead of a vanilla NN.')
  parser.add_argument('-B', '--bottleneck-output', required=False, help='Output file for reduced representation.')
  parser.set_defaults(with_ae=False)
  parser.add_argument('--pretrain-ae', dest='pretrain_ae', action='store_true', help='Pretrains AE then fine tunes the complete model')
  parser.add_argument('-e', '--clf-epochs', type=int, help='Number of epochs for classifier training', default=20)
  parser.add_argument('-E', '--ae-epochs', type=int, help='Number of epochs for AE training', default=20)
  parser.add_argument('-a', '--ae-dims', nargs='*', type=int, action='append', help='List of each layered AE dimensions')
  parser.add_argument('-b', '--bottleneck', nargs='+', type=int, help='Dimensions of each bottleneck layer (i.e., the compressed representation)')
  parser.add_argument('-c', '--clf-dims', nargs='+', type=int, help='List of dense dimensions for classification')
  parser.add_argument('-r', '--cv-round', type=int, help='Iteration number of cross validation.', default=0)
  parser.add_argument('-l', '--loss', choices=['mse', 'mae', 'l1l2'], default='mse')
  parser.add_argument('-S', '--batch-size', type=int, help='Batch size.', default=16)
  parser.add_argument('-f', '--format', choices=['libsvm', 'tecbench'], default='libsvm')
  parser.add_argument('-P', '--path', help='Path with TecBench data files.')

  parser.set_defaults(with_ae=False)
  parser.set_defaults(pretrain_ae=False)

  args = parser.parse_args()

  params = {
     'libsvm_data_handler': {'train': args.train, 'test': args.test, 'val': args.val} if args.format == 'libsvm' else None,
     'tecbench_data_handler': {'path': args.path, 'fold': args.cv_round} if args.format == 'tecbench' else None
  }
  le, x_trn, y_trn, x_tst, y_tst, x_val, y_val = parse(**params)
  #le, x_trn, y_trn, x_tst, y_tst, x_val, y_val = parse(args.train, args.test, args.val, args.format)
  validation_data = (x_val, y_val) if  args.val and x_trn.shape[0] > 0 else None

  loss = l1l2 if args.loss == 'l1l2' else args.loss

  model, ae_model, compressor = build_model(x_trn.shape[1], len(le.classes_), with_ae=args.with_ae, all_ae_dims=args.ae_dims, bottleneck_dims=args.bottleneck, clf_dims=args.clf_dims, loss=loss)
  model, h = fit(model, x_trn, y_trn, validation_data=validation_data, clf_epochs=args.clf_epochs, ae_epochs=args.ae_epochs, with_ae=args.with_ae,
                 pretrain_ae=args.pretrain_ae, batch_size=args.batch_size, use_tb_callback=True)

  if validation_data:
    model, _ = fit(model, validation_data[0], validation_data[1], clf_epochs=args.clf_epochs, ae_epochs=args.ae_epochs, with_ae=args.with_ae, pretrain_ae=args.pretrain_ae, batch_size=args.batch_size)

  if args.with_ae:
    if args.pretrain_ae:
      ae_model.trainable = True
      callback = tf.keras.callbacks.EarlyStopping(monitor='val_glb_classifier_loss' if validation_data and args.with_ae else ('glb_classifier_loss' if args.with_ae else ('val_loss' if validation_data else 'loss')), patience=5, min_delta=0.01)
      tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs-model-ft-trn")
      model.fit(x_trn, y_trn, validation_data=validation_data, batch_size=args.batch_size, epochs=args.clf_epochs, callbacks=[callback, tensorboard_callback])
      callback = tf.keras.callbacks.EarlyStopping(monitor='val_glb_classifier_loss' if validation_data and args.with_ae else ('glb_classifier_loss' if args.with_ae else ('val_loss' if validation_data else 'loss')), patience=5, min_delta=0.01)
      tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs-model-ft-val")
      model.fit(validation_data[0], validation_data[1], batch_size=args.batch_size, epochs=10, callbacks=[callback, tensorboard_callback])
    y_prd = model.predict(x_tst, batch_size=1)[-1]
  else:
    y_prd = model.predict(x_tst, batch_size=1)

  with io.open(args.output, 'a') as fh:
    preds = le.inverse_transform(np.argmax(y_prd, axis=1))
    scores = np.max(y_prd, axis=1)
    fh.write('#{}\n'.format(args.cv_round))
    for i, y_t in enumerate(y_tst):
      fh.write('{} {} {}:{}\n'.format(i, y_t, preds[i], scores[i]))

  if args.bottleneck_output and compressor is not None:
    x_new = compressor.predict(x_trn)
    if validation_data:
      x_new = np.concatenate((x_new, compressor.predict(x_val)), axis=0)
    y = y_trn if not validation_data else np.concatenate((y_trn,y_val), axis=0)
    with io.open('{}_trn'.format(args.bottleneck_output), 'w') as fh:
      for i, x in enumerate(x_new):
        fh.write('{} {}\n'.format(y[i], ' '.join('{}:{}'.format(j+1, v) for j, v in enumerate(x))))
    x_new = compressor.predict(x_tst)
    with io.open('{}_tst'.format(args.bottleneck_output), 'w') as fh:
      for i, x in enumerate(x_new):
        fh.write('{} {}\n'.format(y_tst[i], ' '.join('{}:{}'.format(j+1, v) for j, v in enumerate(x))))

  if args.plot:
    plot(model, h, args.plot)

  if args.save_model:
     if args.with_ae:
       pickle.dump('{}_encoder'.format(args.save_model))
       if compressor is not None:
         compressor.save('{}_compressor'.format(args.save_model))
     model.save('{}_model'.format(args.save_model))


