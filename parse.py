import argparse
import io
import csv
import scipy
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.utils import shuffle


def add_data(r, indptr, indices, data, vocab):
  if len(r) > 1:
    label = int(r[0])
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

  y = np.array(y, dtype=np.uint8)
  return y, indptr, indices, data, vocab


def parse(trn_fn, tst_fn, val_data=None):
  val_fn, val_split = None, None
  if val_data:
    try:
      val_split = float(val_data)
      assert val_split > 0 and val_split < 1, 'Validation split must be on the (0 , 1) open interval.'
    except:
      val_fn = val_data

  indptr = [0]
  indices, data, vocab = [], [], dict()

  y_trn, indptr, indices, data, vocab = process_file(trn_fn, indptr, indices, data, vocab)
  y_tst, indptr, indices, data, vocab = process_file(tst_fn, indptr, indices, data, vocab) 

  labels = set(y_trn)

  x_val, y_val = None, None
  if val_fn:
    y_val, indptr, indices, data, vocab = process_file(val_fn, indptr, indices, data, vocab)

  x =  csr_matrix((data, indices, indptr), dtype=np.float32)
  x.sort_indices()

  x_trn = x[:len(y_trn)]
  x_trn, y_trn = shuffle(x_trn, y_trn)

  x_tst = x[len(y_trn):]

  if val_fn:
    x_val = x_trn[len(y_trn)+len(y_tst):]
    x_trn = x_trn[:len(y_trn)]
  elif val_split:
    sz = int(len(y_trn) * (1.0 - val_split))
    x_val = x_trn[sz:len(y_trn)]
    x_trn = x_trn[:sz]
    y_val = y_trn[sz:]
    y_trn = y_trn[:sz]

#  x_trn, y_trn = shuffle(x_trn, y_trn)
#  if x_val is not None:
#    x_val, y_val = shuffle(x_val, y_val)

#  y_trn = []
#  with io.open(trn_fn) as fh:
#    csvr = csv.reader(fh, delimiter = ' ')
#    indptr = [0]
#    indices, data, vocab = [], [], dict()
#    for r in csvr:
#      label, indptr, indices, data, vocab = add_data(r, indptr, indices, data, vocab)
#      if label is not None:
#        y_trn.append(label)
#
#  x_trn = csr_matrix((data, indices, indptr), dtype=np.float32)
#  x_trn.sort_indices()
#
#  y_trn = np.array(y_trn, dtype=np.uint8)
#
#  y_tst = []
#  with io.open(tst_fn) as fh:
#    csvr = csv.reader(fh, delimiter = ' ')
#    for r in csvr:
#      label, indptr, indices, data, vocab = add_data(r, indptr, indices, data, vocab)
#      if label is not None:
#        y_tst.append(label)
#  x_tst = csr_matrix((data, indices, indptr), dtype=np.float32)[len(y_trn):]
#  x_tst.sort_indices()
#
#  y_tst = np.array(y_tst, dtype=np.uint8)

  return labels, x_trn, y_trn, x_tst, y_tst, x_val, y_val


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Parses a libSVM-formatted dataset.')
  parser.add_argument('-t', '--train', required=True, help='Training file in libSVM format.')
  parser.add_argument('-v', '--val', required=False, help='Validation file in libSVM format or a validation split rate on (0,1) open interval.')
  parser.add_argument('-T', '--test', required=True, help='Test file in libSVM format.')

  args = parser.parse_args()

  # X, y = sklearn.utils.shuffle(X, y)
    
  labels, x_trn, y_trn, x_tst, y_tst, x_val, y_val = parse(args.train, args.test, args.val)

  print(len(labels))
  print(x_trn.shape, len(y_trn))
  print(x_tst.shape, len(y_tst))
  if args.val:
    print(x_val.shape, len(y_val))
