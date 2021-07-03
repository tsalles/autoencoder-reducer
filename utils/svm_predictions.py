import argparse
import io
import csv


def parse(prd_fn, tst_fn):
  y = []
  with io.open(tst_fn) as fh:
    csvr = csv.reader(fh, delimiter = ' ')
    for r in csvr:
      if len(r) > 1:
        y.append(r[0].strip())

  y_p = []
  with io.open(prd_fn) as fh:
    for l in fh:
      if l:
        y_p.append(int(l.strip()))
  
  return y, y_p


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Parses a libSVM-formatted dataset.')
  parser.add_argument('-i', '--input', required=True, help='SVM predictions file.')
  parser.add_argument('-t', '--test', required=True, help='Coresponding test file in libSVM format.')
  parser.add_argument('-o', '--output', required=True, help='Output file with real class and predicted class.')
  parser.add_argument('-r', '--cv-round', required=False, type=int, help='Marker for defining the beginning of a cross validation round.')

  args = parser.parse_args()

  y, y_p = parse(args.input, args.test)
  with io.open(args.output, 'a') as fh:
    fh.write('#{}\n'.format(args.cv_round))
    for i, (r, p) in enumerate(zip(y, y_p)):
      fh.write('{} {} {}:1\n'.format(i, r, p))


