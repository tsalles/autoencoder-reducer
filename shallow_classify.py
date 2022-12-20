import sys
import io
import scipy.sparse as sps

from preprocessing.parse import parse
from sklearn.datasets import load_svmlight_file

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

import argparse


def fit(clf, X, y, params):
#  grd = GridSearchCV(clf, params)
  grd = RandomizedSearchCV(clf, params, n_iter=10, cv=3)
  return grd.fit(X, y)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Trains and classifies textual data in TFIDF representation using XGBoost.')
  parser.add_argument('-t', '--train', help='LibSVM train file.')
  parser.add_argument('-T', '--test', help='LibSVM test file.')
  parser.add_argument('-r', '--cv-round', type=int, help='Iteration number of cross validation.', default=0)
  parser.add_argument('-m', '--method', choices=['xb', 'rf', 'xt'], default='xb')
  parser.add_argument('-o', '--output', help='Output prediction file.')
#  parser.add_argument('-v', '--val', required=False, help='Validation file in libSVM format or a validation split on (0,1) open interval.')

  args = parser.parse_args()

  x_trn, y_trn = load_svmlight_file(args.train)
  x_tst, y_tst = load_svmlight_file(args.test)
  x_tst = sps.csr_matrix((x_tst.data, x_tst.indices, x_tst.indptr), shape=(x_tst.shape[0], x_trn.shape[1]))

  model, params = None, dict()
  if args.method == 'xb':
    model = xgb.XGBClassifier(tree_method='gpu_hist',
                              eval_metric='mlogloss',
                              use_label_encoder=False) # eta=0.01, gamma=0.001, max_depth=8, subsample=0.7
    params = {
      'min_child_weight': [1, 5, 10],
      'gamma': [0.001, 0.1, 1],
      'subsample': [0.6, 0.8, 1.0],
      'colsample_bytree': [0.6, 0.8, 1.0],
      'max_depth': [3, 5, 7]
    }
  elif args.method == 'rf':
    model = RandomForestClassifier(n_jobs=-1)
    params = {
      'n_estimators': [100, 500, 1000],
      'max_features': ['log2', 'sqrt'],
      'max_depth': [10, 100, 300],
    }
  elif args.method == 'xt':
    model = ExtraTreesClassifier(n_jobs=-1)
    params = {
      'n_estimators': [100, 500, 1000],
      'max_features': ['log2', 'sqrt'],
      'max_depth': [10, 100, 300],
    }
  else:
    print('Not supported')
    sys.exit(1)

  clf = fit(model, x_trn, y_trn, params)
  y_pred = clf.predict(x_tst)
  preds = [round(value) for value in y_pred]
  with io.open(args.output, 'a') as fh:
    fh.write('#{}\n'.format(args.cv_round))
    for i, (p, y) in enumerate(zip(preds, y_tst)):
        fh.write('{} {} {}:1\n'.format(i, int(y), int(p)))

  accuracy = accuracy_score(y_tst, preds)
  print("Accuracy: %.2f%%" % (accuracy * 100.0))


