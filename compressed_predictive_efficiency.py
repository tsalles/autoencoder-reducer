import argparse
import numpy as np
from sklearn.metrics import f1_score


def parse(fn):
  feats = set()
  with open(fn) as fh:
    for l in fh:
      l = l.strip()
      flds = l.split(' ')
      for f in flds:
        feats.add(f.split(':')[0])
  return len(feats)


def compute_f1(fn):
  mic_f1, mac_f1 = [], []
  with open(fn) as fh:
    real, preds = [], []
    for x in fh:
      x = x.strip()
      if len(x) == 2 and x[0] == '#':
        if preds and real and len(preds) == len(real):
          print('Appending f1 score for current round')
          mic_f1.append(f1_score(real, preds, average='micro'))
          mac_f1.append(f1_score(real, preds, average='macro'))
        preds, real = [], []
      elif len(x.split(' ')) > 1:
        preds.append(x.split(' ')[-1].split(':')[0])
        real.append(x.split(' ')[1])
  # Work on last observed round
  if preds and real and len(preds) == len(real):
    mic_f1.append(f1_score(real, preds, average='micro'))
    mac_f1.append(f1_score(real, preds, average='macro'))
  return mic_f1, mac_f1


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Paired t-test for two prediction results for Macro and Micro F1.')
  parser.add_argument('-t', '--train', required=True, help='Original training files in LibSVM format.')
  parser.add_argument('-f', '--folds-path', required=True, help='Path of folder with original training files in LibSVM format.')
  parser.add_argument('-r', '--reduced-dimension', type=int, required=False, help='Reduced dimension. If not specified equals to number of features in training set.')
  parser.add_argument('-R', '--cv-rounds', type=int, required=True, help='Number of CV rounds.')
  parser.add_argument('-p', '--predictions', required=True, help='Prediction file.')

  args = parser.parse_args()

  mic_f1, mac_f1 = compute_f1(args.predictions)

  mic_cpes = []
  mac_cpes = []
  for i in range(0, args.cv_rounds):
    n_features = float(parse('{}/fold_{}/{}'.format(args.folds_path, i, args.train)))
    n_reduced = args.reduced_dimension if args.reduced_dimension else n_features
    reduction_rate = 1.0 - (n_reduced/n_features)
    mic_predictive_efficiency = mic_f1[i]
    mac_predictive_efficiency = mac_f1[i]
    mic_cpe = 2 * reduction_rate * mic_predictive_efficiency / (reduction_rate + mic_predictive_efficiency)
    mac_cpe = 2 * reduction_rate * mac_predictive_efficiency / (reduction_rate + mac_predictive_efficiency)
    mic_cpes.append(mic_cpe)
    mac_cpes.append(mac_cpe)
  print('{},{}'.format(np.mean(mic_cpe), np.mean(mac_cpe)))

