import sys
import io
import numpy as np
from scipy import sparse
from preprocessing.parse import parse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD, PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif


import argparse


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Trains and classifies textual data using shallow classifiers.')
  parser.add_argument('-P', '--path', help='Path with TecBench data files.')
  parser.add_argument('-r', '--cv-round', type=int, help='Iteration number of cross validation.', default=0)
  parser.add_argument('-d', '--dims', type=int, help='Number of dimensions to reduce.', default=256)
  parser.add_argument('-m', '--method', choices=['svd', 'nmf', 'chi', 'mic'], default='svd')
  parser.add_argument('-o', '--output', help='Output prediction file.')
#  parser.add_argument('-v', '--val', required=False, help='Validation file in libSVM format or a validation split on (0,1) open interval.')

  args = parser.parse_args()
  params = {
      'tecbench_data_handler': {'path': args.path, 'fold': args.cv_round}
  }

  le, x_trn, y_trn, x_tst, y_tst, x_val, y_val = parse(**params)

  transform_test = True
  if args.method == 'svd':
    dec = TruncatedSVD(n_components=args.dims)
#  elif args.method == 'pca':
#    dec = PCA(n_components=args.dims, iterated_power=7)
  elif args.method == 'nmf':
    dec = NMF(n_components=args.dims)
  elif args.method == 'chi':
    dec = SelectKBest(chi2, k=args.dims)
    transform_test = False
  elif args.method == 'mic':
    dec = SelectKBest(mutual_info_classif, k=args.dims)
    transform_test = False
  else:
    print('Not supported')
    sys.exit(1)

  normalizer = Normalizer(copy=False)
  red = make_pipeline(dec, normalizer)
  X = red.fit_transform(x_trn, y_trn)

  with io.open('{}_trn'.format(args.output), 'w') as fh:
    if sparse.issparse(X):
        X = [x.tolist()[0] for x in X.todense()]
    for x, y in zip(X, y_trn):
      feats = ' '.join(['{}:{}'.format(k+1, v) for k, v in enumerate(x) if np.any(v)]).strip()
      if feats:
        fh.write('{} {}\n'.format(y, feats))

  X = red.transform(x_tst) if transform_test else X
  with io.open('{}_tst'.format(args.output), 'w') as fh:
    if sparse.issparse(X):
        X = [x.tolist()[0] for x in X.todense()]
    for x, y in zip(X, y_tst):
      feats = ' '.join(['{}:{}'.format(k+1, v) for k, v in enumerate(x) if np.any(v)]).strip()
      if feats:
        fh.write('{} {}\n'.format(y, feats))

