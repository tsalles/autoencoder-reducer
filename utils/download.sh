#!/bin/bash

curl http://hidra.lbd.dcc.ufmg.br/newdatasets/ -o index.html
for dataset in `xmllint --html --xpath '/html/body/table/tr/td/a/node()' index.html`; do
  if [[ $dataset != "Parent"  && $dataset != "Directory" ]]; then
    echo $dataset;
    mkdir -p $dataset/10fold $dataset/5fold;
    curl http://hidra.lbd.dcc.ufmg.br/newdatasets/${dataset}texts.txt -o $dataset/texts.txt;
    curl http://hidra.lbd.dcc.ufmg.br/newdatasets/${dataset}score.txt -o $dataset/score.txt;
    for f in 5 10; do
      curl http://hidra.lbd.dcc.ufmg.br/newdatasets/${dataset}${f}fold/tfidf/ -o folds_idx.txt;
      for filename in `xmllint --html --xpath '/html/body/table/tr/td/a/node()' folds_idx.txt`; do
        if [[ $filename != "Parent"  && $filename != "Directory" ]]; then
          curl http://hidra.lbd.dcc.ufmg.br/newdatasets/${dataset}split_${f}.csv -o $dataset/split_${f}.csv;
          curl http://hidra.lbd.dcc.ufmg.br/newdatasets/${dataset}${f}fold/tfidf/$filename -o $dataset/${f}fold/$filename;
        fi;
      done;
    done;
    rm folds_idx.txt;
  fi;
done

rm index.html;
