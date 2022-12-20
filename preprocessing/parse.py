import sys
import argparse
import io
import csv
import scipy
from scipy.sparse import csr_matrix
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import dump_svmlight_file
import pickle
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
try:
  from preprocessing.datamodule.TecDataModule import TeCDataModule
except:
  from datamodule.TecDataModule import TeCDataModule
import torch

torch.multiprocessing.set_sharing_strategy('file_system')

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


def process_file(fn,  indptr, indices, data, vocab, samples=None):
  y = []
  with io.open(fn) as fh:
    csvr = csv.reader(fh, delimiter = ' ')
    for r in csvr:
      label, indptr, indices, data, vocab = add_data(r, indptr, indices, data, vocab)
      if label is not None:
        y.append(label)

  return y, indptr, indices, data, vocab


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def parse(libsvm_data_handler=None, tecbench_data_handler=None):
  if not libsvm_data_handler and not tecbench_data_handler:
    raise RuntimeError('Please provide either libSVM or TecBench parameters.')
  if ((libsvm_data_handler and not (libsvm_data_handler.get('trn_fn', None) and libsvm_data_handler.get('tst_fn'))) or
      (tecbench_data_handler and not (tecbench_data_handler.get('path', None) and tecbench_data_handler.get('fold', None) is not None))):
    raise RuntimeError('LibSVM: train and test files must be specified.' if libsvm_data_handler else 'TeCBench: path and folder must be specified.')

  if libsvm_data_handler:
    return parse_libsvm(**libsvm_data_handler)
  else:
    if tecbench_data_handler and tecbench_data_handler.get('path', None) and tecbench_data_handler['path'][-1] != '/':
      tecbench_data_handler['path'] = tecbench_data_handler['path'] + '/'
    return parse_tecbench(**tecbench_data_handler)


def parse_tecbench(path, fold):
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  params = {
    'dir': path,
    'max_length': 128,
    'batch_size': 8,
    'num_workers': 0
  }
  data_gen = TeCDataModule(AttrDict(params), tokenizer, fold=fold)
  data_gen.prepare_data()
  data_gen.setup(stage='fit')

  vectorizer = TfidfVectorizer(min_df=5, max_df=.2, max_features=50000)
  le = LabelEncoder()

  xs, ys = zip(*[(' '.join(['t{}'.format(i) for i in x]), y) for b in data_gen.train_dataloader() for x, y in zip(b['text'], b['cls'])])
  x_trn = csr_matrix(vectorizer.fit_transform(xs))
  x_trn.sort_indices()
  y_trn = np.array(le.fit_transform(ys))


  xs, ys = zip(*[(' '.join(['t{}'.format(i) for i in x]), y) for b in data_gen.val_dataloader() for x, y in zip(b['text'], b['cls'])])
  x_val = csr_matrix(vectorizer.transform(xs))
  x_val.sort_indices()
  y_val = np.array(le.transform(ys))

  data_gen.setup(stage='test')
  xs, ys = zip(*[(' '.join(['t{}'.format(i) for i in x]), y) for b in data_gen.test_dataloader() for x, y in zip(b['text'], b['cls'])])
  x_tst = csr_matrix(vectorizer.transform(xs))
  x_tst.sort_indices()
  y_tst = np.array(le.transform(ys))

  return le, x_trn, y_trn, x_tst, y_tst, x_val, y_val



def parse_libsvm(trn_fn, tst_fn, val_data=None):
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

  le = LabelEncoder()

  x_val, y_val = None, None
  if val_fn:
    y_val, indptr, indices, data, vocab = process_file(val_fn, indptr, indices, data, vocab)
    le = le.fit(y_trn+y_val)
    y_trn = le.transform(y_trn)
    y_val = le.transform(y_val)
  else:
    y_trn = le.fit_transform(y_trn)

  x =  csr_matrix((data, indices, indptr), dtype=np.float32)
  x.sort_indices()

  x_tst = x[len(y_trn):]

  x_trn = None
  if not val_fn and not val_split:
    x_trn, y_trn = shuffle(x[:len(y_trn)], y_trn)
  else:
    if val_fn:
      x_trn = x[:len(y_trn)+len(y_tst)]
      x_trn, y_trn = shuffle(x_trn[:len(y_trn)+len(y_tst)], y_trn+y_val)
      x_val = x_trn[len(y_val):]
      x_trn = x_trn[:len(y_val)]
    else:
      sz = int(len(y_trn) * (1.0 - val_split))
      x_trn, y_trn = shuffle(x[:len(y_trn)], y_trn)
      x_val = x_trn[sz:]
      x_trn = x_trn[:sz]
      y_val = y_trn[sz:]
      y_trn = y_trn[:sz]

  return le, x_trn, y_trn, x_tst, y_tst, x_val, y_val


def write_svm(X, y, prefix, suffix):
  with io.open('{}_{}.svm'.format(prefix, suffix), 'wb') as fh:
    dump_svmlight_file(X, y, fh, zero_based=False)


#  with io.open('{}_{}.svm'.format(prefix, suffix), 'w', encoding='utf8') as fh:
#    for i, x in enumerate(X):
#      ln = '{}'.format(y[i])
#      for f, v in enumerate(x):
#        ln += ' {}:{}'.format(f, v)
#      ln += '\n'
#      fh.write(ln)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Parses a libSVM-formatted dataset.')
  parser.add_argument('-t', '--train', help='Training file in libSVM format.')
  parser.add_argument('-v', '--val', help='Validation file in libSVM format or a validation split rate on (0,1) open interval.')
  parser.add_argument('-T', '--test', help='Test file in libSVM format.')
  parser.add_argument('-p', '--path', help='Path with TecBench data files.')
  parser.add_argument('-F', '--fold', help='Fold number for TecBench parser.')
  parser.add_argument('-f', '--format', choices=['libsvm', 'tecbench'], default='libsvm')
  parser.add_argument('-o', '--output', help='Outputs the same data in libSVM format with the given prefix.')
  args = parser.parse_args()

  # X, y = sklearn.utils.shuffle(X, y)

  params = {
    'libsvm_data_handler': {'train': args.train, 'test': args.test, 'val': args.val} if args.format == 'libsvm' else None,
    'tecbench_data_handler': {'path': args.path, 'fold': args.fold} if args.format == 'tecbench' else None
  }
  le, x_trn, y_trn, x_tst, y_tst, x_val, y_val = parse(**params)

  if args.output:
    write_svm(x_trn, y_trn, args.output, 'trn')
    write_svm(x_tst, y_tst, args.output, 'tst')
    if args.val:
      write_svm(x_val, y_val, args.output, 'val')
