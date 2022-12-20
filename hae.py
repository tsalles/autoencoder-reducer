import tensorflow as tf
import time
import argparse
import io
import numpy as np

from preprocessing.parse import parse


class Classifier(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, activation='relu'):
        super(Classifier, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(input_dim, activation=activation)
        self.output_layer = tf.keras.layers.Dense(output_dim, activation='softmax')

    def call(self, input_tensor, training=False):
        x = self.hidden_layer(input_tensor)
        y = self.output_layer(x)
        return y


class AutoEncoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, bottleneck_dim, ae_dims, activation='relu'):
        super(AutoEncoder, self).__init__()
        self.encoders = [tf.keras.layers.Dense(input_dim, activation=activation)]
        self.encoders += [tf.keras.layers.Dense(dim, activation=activation) for dim in ae_dims]
        self.bottleneck = tf.keras.layers.Dense(bottleneck_dim, activation=activation,
                                                activity_regularizer=tf.keras.regularizers.L2(0.01),
                                                kernel_regularizer=tf.keras.regularizers.OrthogonalRegularizer(factor=0.01, mode="columns"))
        self.decoders = [tf.keras.layers.Dense(dim, activation=activation) for dim in reversed(ae_dims)]
        self.decoders.append(tf.keras.layers.Dense(input_dim, activation='sigmoid'))

    def call(self, input_tensor, training=False):
        x = None
        for enc in self.encoders:
            x = enc(input_tensor) if x is None else enc(x)
            x = tf.keras.layers.Dropout(0.2)(x)
        reduced = tf.keras.layers.Dropout(0.2)(self.bottleneck(x))
        x = reduced
        for dec in self.decoders:
            x = dec(x)
        reconstructed = x
        return reconstructed, reduced


class HAE(tf.keras.Model):
    def __init__(self, input_dim, hae_dims, bottleneck_dims, num_classes, ae_activation='relu', supervised=True):
        super(HAE, self).__init__()
        self.autoencoders = []
        self.classifiers = [] if supervised else None
        for level, ae_dims in enumerate(hae_dims):
            self.autoencoders.append(AutoEncoder(input_dim, bottleneck_dims[level], ae_dims, ae_activation))
            if supervised:
                self.classifiers.append(Classifier(bottleneck_dims[level], num_classes))
            input_dim = bottleneck_dims[level]
        #self.hidden = tf.keras.layers.Dense(64, activation='relu')
        self.clf = tf.keras.layers.Dense(num_classes, activation='softmax')


    def call(self, input_tensor, training=False, **kwargs):
        # forward pass
        current_input = input_tensor
        #current_input = input_tensor
        reconstructed, reduced = None, None
        pairs = []
        for level, ae in enumerate(self.autoencoders):
            reconstructed, reduced = ae(current_input)
            partial = (current_input, reconstructed)
            # call clf and append prediction within the below tuple
            if self.classifiers is not None:
                pred_y = self.classifiers[level](reduced)
                partial += (pred_y,)
            pairs.append(partial)
            current_input = reduced
        current_input = tf.keras.layers.Concatenate()([r[1] for r in pairs])
        pred = self.clf(current_input) #self.hidden(current_input))
        return pairs, reduced, pred


    def build_graph(self, raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))



#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#x_train = x_train.reshape(60000, 784).astype("float32") / 255
#train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#train_dataset = train_dataset.shuffle(buffer_size=2048).batch(256)



def fit(train_dataset, val_dataset, epochs, hae, alpha):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    mse_loss_fn = tf.keras.losses.MeanAbsoluteError()
    sce_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    train_loss_metric = tf.keras.metrics.Mean()
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            pairs, reduced, preds = hae(x)
            # Compute reconstruction loss for each level
            sce_loss = sce_loss_fn(y, preds)# * len(pairs)
            if len(pairs[0]) == 3:
                local_sce_loss = sum([sce_loss_fn(y, pair[2])
                                      for i, pair in enumerate(pairs)]) / len(pairs)
                sce_loss += local_sce_loss
                sce_loss /= 2.0
            mse_loss = sum([mse_loss_fn(tf.sparse.to_dense(pair[0]) if i == 0 else pair[0], pair[1])
                            for i, pair in enumerate(pairs)]) / len(pairs)
            #loss = (2 * sce_loss * mse_loss) / (sce_loss + mse_loss)
            loss = alpha*sce_loss + (1-alpha)*mse_loss
            #for pair in pairs:
            #    loss += mse_loss_fn(pair[0], pair[1])
        grads = tape.gradient(loss, hae.trainable_weights)
        optimizer.apply_gradients(zip(grads, hae.trainable_weights))
        train_loss_metric(loss)
        train_acc_metric.update_state(y, preds)
        return train_loss_metric.result()

    val_loss_metric = tf.keras.metrics.Mean()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def val_step(x, y):
        pairs, reduced, preds = hae(x)

        sce_loss = sce_loss_fn(y, preds)# * len(pairs)
        if len(pairs[0]) == 3:
            local_sce_loss = sum([sce_loss_fn(y, pair[2])
                                  for i, pair in enumerate(pairs)]) / len(pairs)
            sce_loss += local_sce_loss
            sce_loss /= 2.0
        mse_loss = sum([mse_loss_fn(tf.sparse.to_dense(pair[0]) if i == 0 else pair[0], pair[1])
                        for i, pair in enumerate(pairs)]) / len(pairs)
        #loss = (2 * sce_loss * mse_loss) / (sce_loss + mse_loss)
        loss = alpha*sce_loss + (1-alpha)*mse_loss

        val_loss_metric(loss)
        val_acc_metric.update_state(y, preds)
        return val_loss_metric.result()

    epochs = 100
    patience = 10
    tol = 0.00001
    wait = 0
    min_loss = None

    # Iterate over epochs.
    for epoch in range(epochs):
        t = time.time()
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            train_loss_value = train_step(x_batch_train, y_batch_train)

        for val_batch_step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = val_step(x_batch_val, y_batch_val)

        wait += 1
        if min_loss is None or (min_loss - val_loss_value) > tol:
            min_loss = val_loss_value
            wait = 0

        template = 'ETA: {} - epoch: {} loss: {}  acc: {} val_loss: {} val_acc: {} best_loss: {} wait: {}'
        print(template.format(
              round((time.time() - t)/60, 2), epoch + 1,
              train_loss_value, float(train_acc_metric.result()),
              val_loss_value, float(val_acc_metric.result()),
              min_loss, wait
        ))

        train_acc_metric.reset_states()

        if wait >= patience:
            print('Early Stopping: {} epochs without improvements on validation loss.'.format(wait))
            break
    return hae



#x_test = x_test.reshape(10000, 784).astype("float32") / 255
#test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
#test_dataset = test_dataset.shuffle(buffer_size=2048).batch(256)


def predict(test_dataset, hae, alpha, out_fn, reduced_fn):
    mse_loss_fn = tf.keras.losses.MeanAbsoluteError()
    sce_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    test_loss_metric = tf.keras.metrics.Mean()
    test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def test_step(x, y):
        pairs, reduced, preds = hae(x)

        sce_loss = sce_loss_fn(y, preds)# * len(pairs)
        if len(pairs[0]) == 3:
            local_sce_loss = sum([sce_loss_fn(y, pair[2])
                                  for i, pair in enumerate(pairs)]) / len(pairs)
            sce_loss += local_sce_loss
            sce_loss /= 2.0
        mse_loss = sum([mse_loss_fn(tf.sparse.to_dense(pair[0]) if i == 0 else pair[0], pair[1])
                        for i, pair in enumerate(pairs)]) / len(pairs)
        #loss = (2 * sce_loss * mse_loss) / (sce_loss + mse_loss)
        loss = alpha*sce_loss + (1-alpha)*mse_loss

        test_loss_metric(loss)
        test_acc_metric.update_state(y, preds)
        return test_loss_metric.result(), preds, reduced

    with io.open(out_fn, 'a') as fh, io.open(reduced_fn, 'a') as rfh:
        fh.write('#{}\n'.format(args.cv_round))
        for step, (x_batch_test, y_batch_test) in enumerate(test_dataset):
            test_loss_value, preds, reduced = test_step(x_batch_test, y_batch_test)
            y_prd = le.inverse_transform(np.argmax(preds, axis=1))
            scores = np.max(preds, axis=1)
            for i, y_t in enumerate(y_batch_test):
                fh.write('{} {} {}:{}\n'.format(i, y_t, y_prd[i], scores[i]))
                rfh.write('{} {} {}'.format(i, y_batch_test[i], ' '.join(['{}:{}'.format(f, v) for f, v in enumerate(reduced[i])])))
    print('Test Loss', float(test_loss_value))
    print('Test Accuracy', float(test_acc_metric.result()))

#hae.summary()
#



def build_model(input_dim, batch_size, hae_dims, bottleneck_dims, num_classes):
    # init model object

    x = tf.keras.Input(batch_size=batch_size, shape=(input_dim,), sparse=True)
    hae = HAE(input_dim, hae_dims, bottleneck_dims, num_classes)

    return tf.keras.Model(inputs=x, outputs=hae(x))



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Hierarchical Autoencoder for dimensionality reduction.')
  parser.add_argument('-t', '--train', help='Training file in libSVM format.')
  parser.add_argument('-T', '--test', help='Test file in libSVM format.')
  parser.add_argument('-v', '--val', help='Validation file in libSVM format or a validation split on (0,1) open interval.')
  parser.add_argument('-o', '--output', required=True, help='Output file for predictions.')
  parser.add_argument('-s', '--save-model', required=False, help='Prefix file for storing trained model.')
  parser.add_argument('-B', '--bottleneck-output', required=False, help='Output file for reduced representation.')
  parser.add_argument('-E', '--epochs', type=int, help='Number of epochs for AE training', default=20)
  parser.add_argument('-a', '--ae-dims', nargs='*', type=int, action='append', help='List of each layered AE dimensions')
  parser.add_argument('-b', '--bottleneck', nargs='+', type=int, help='Dimensions of each bottleneck layer (i.e., the compressed representation)')
#  parser.add_argument('-c', '--clf-dims', nargs='+', type=int, help='List of dense dimensions for classification')
  parser.add_argument('-r', '--cv-round', type=int, help='Iteration number of cross validation.', default=0)
  parser.add_argument('-l', '--loss', choices=['mse', 'mae', 'l1l2'], default='mse')
  parser.add_argument('-S', '--batch-size', type=int, help='Batch size.', default=16)
  parser.add_argument('-f', '--format', choices=['libsvm', 'tecbench'], default='libsvm')
  parser.add_argument('-P', '--path', help='Path with TecBench data files.')

  args = parser.parse_args()

  params = {
     'libsvm_data_handler': {'trn_fn': args.train, 'tst_fn': args.test, 'val_data': args.val} if args.format == 'libsvm' else None,
     'tecbench_data_handler': {'path': args.path, 'fold': args.cv_round} if args.format == 'tecbench' else None
  }
  le, x_trn, y_trn, x_tst, y_tst, x_val, y_val = parse(**params)
  validation_data = True if args.val and x_trn.shape[0] > 0 else False

  loss = l1l2 if args.loss == 'l1l2' else args.loss

  alpha = 0.5

  hae_dims = args.ae_dims # [[256, 128], [32, 16]]
  bottleneck_dims = args.bottleneck # [64, 8]
  batch_size = 16

  hae = build_model(x_trn.shape[-1], batch_size, hae_dims, bottleneck_dims, len(set(y_trn)))
#  tf.keras.utils.plot_model(hae.build_graph((x_trn.shape[-1])), to_file="model.png", expand_nested=True, show_shapes=True)
  x_trn = tf.SparseTensor(indices=np.vstack([*x_trn.nonzero()]).T, values=x_trn.data, dense_shape=x_trn.shape)

  trn_dataset = tf.data.Dataset.from_tensor_slices((x_trn, y_trn))
  trn_dataset = trn_dataset.shuffle(buffer_size=2048).batch(batch_size)

  val_dataset = None
  if validation_data:
      x_val = tf.SparseTensor(indices=np.vstack([*x_val.nonzero()]).T, values=x_val.data, dense_shape=x_val.shape)
      val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
      val_dataset = val_dataset.shuffle(buffer_size=2048).batch(batch_size)

  hae = fit(trn_dataset, val_dataset, args.epochs, hae, alpha=alpha)

  x_tst = tf.SparseTensor(indices=np.vstack([*x_tst.nonzero()]).T, values=x_tst.data, dense_shape=x_tst.shape)

  tst_dataset = tf.data.Dataset.from_tensor_slices((x_tst, y_tst))
  tst_dataset = tst_dataset.shuffle(buffer_size=2048).batch(batch_size)
  predict(tst_dataset, hae, alpha, args.output, args.bottleneck_output)

