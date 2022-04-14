from scipy import stats
from sklearn.metrics import f1_score

import argparse



def compute_scores(preds_fn_a, preds_fn_b):
  mic_a, mac_a = [], []
  mic_b, mac_b = [], []
  with open(preds_fn_a) as f1, open(preds_fn_b) as f2:
    real_a, real_b = [], []
    preds_a, preds_b = [], []
    for x, y in zip(f1, f2):
      x, y = x.strip(), y.strip()
      if len(x) == 2 and len(y) == 2 and x[0] == '#' and y[0] == '#':
        assert x[1] == y[1], 'Round mismatch: {} <-> {}'.format(x, y)
        if preds_a and preds_b and len(preds_a) == len(preds_b):
          print('Appending f1 score for current round')
          mic_a.append(f1_score(real_a, preds_a, average='micro'))
          mac_a.append(f1_score(real_a, preds_a, average='macro'))
          mic_b.append(f1_score(real_b, preds_b, average='micro'))
          mac_b.append(f1_score(real_b, preds_b, average='macro'))
        preds_a, preds_b, real_a, real_b = [], [], [], []
      elif len(x.split(' ')) > 1 and len(y.split(' ')) > 1:
        preds_a.append(x.split(' ')[-1].split(':')[0])
        preds_b.append(y.split(' ')[-1].split(':')[0])
        real_a.append(x.split(' ')[1])
        real_b.append(y.split(' ')[1])
  # Work on last observed round
  if preds_a and preds_b and len(preds_a) == len(preds_b):
    print('Appending f1 score for current round')
    mic_a.append(f1_score(real_a, preds_a, average='micro'))
    mac_a.append(f1_score(real_a, preds_a, average='macro'))
    mic_b.append(f1_score(real_b, preds_b, average='micro'))
    mac_b.append(f1_score(real_b, preds_b, average='macro'))
  return (stats.ttest_ind(mic_a, mic_b).pvalue, stats.ttest_ind(mac_a, mac_b).pvalue) if len(mic_a) > 1 else (1, 1)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Paired t-test for two prediction results for Macro and Micro F1.')
  parser.add_argument('-i', '--inputs', nargs=2, required=True, help='Pair of prediction results to be tested.')
  parser.add_argument('-c', '--confidence', type=float, help='Confidence level to be tested against p-value.', default=0.05)

  args = parser.parse_args()

  p_mic, p_mac = compute_scores(*args.inputs)
  print(p_mic, p_mac)
