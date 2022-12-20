#!/bin/bash

for dataset in `ls -d ./*`; do
    pwd;
    echo $dataset;
    for f in 5 10; do
      cd $dataset/${f}fold/;
      pwd;
      for filename in `ls *.gz`; do
        gzip -d $filename;
      done;
      cd -;
    done;
done
