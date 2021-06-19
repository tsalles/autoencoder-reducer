import argparse
import io
import csv
import scipy
from scipy.sparse import csr_matrix
import numpy as np
import tensorflow as tf


def add_data(r, indptr, indices, data, vocab):
  if len(r) > 1:
    label = r[0]
    for f in r[1:]:
      if f:
        k, v = f.split(':')
        idx = vocab.setdefault(k, len(vocab))
        indices.append(idx)
        data.append(float(v))
    indptr.append(len(indices))
    return label, indptr, indices, data, vocab
  return False, indptr, indices, data, vocab


def process_file(fn,  indptr, indices, data, vocab):
  y = []
  with io.open(fn) as fh:
    csvr = csv.reader(fh, delimiter = ' ')
    for r in csvr:
      label, indptr, indices, data, vocab = add_data(r, indptr, indices, data, vocab)
      if label is not None:
        y.append(label)

  return y, indptr, indices, data, vocab


def parse(data_fn):
  indptr = [0]
  indices, data, vocab = [], [], dict()

  y, indptr, indices, data, vocab = process_file(data_fn, indptr, indices, data, vocab)

  x =  csr_matrix((data, indices, indptr), dtype=np.float32)
  x.sort_indices()

  return x, y


def compress(x, y, model, out_fn):
  x_new = model.predict(x)
  with io.open(out_fn, 'w') as fh:
    for i, x in enumerate(x_new):
      fh.write('{} {}\n'.format(y[i], ' '.join('{}:{}'.format(j, v) for j, v in enumerate(x))))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Parses a libSVM-formatted dataset.')
  parser.add_argument('-d', '--dataset', required=True, help='Input dataset for reduction.')
  parser.add_argument('-m', '--model', required=False, help='Trained compressor model file.')
  parser.add_argument('-o', '--output', required=True, help='Output file with reduced data in libSVM format.')

  args = parser.parse_args()

  x, y = parse(args.dataset)

  model = tf.keras.models.load_model(args.model)

  compress(x, y, model, args.output)

