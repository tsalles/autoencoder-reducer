#!/bin/bash

for i in 16 32 64 128 256 512 1024 2048; do
    rm res_nlp_$i;
    for r in `seq 0 9`; do
        python3 model_layered_clf.py -f tecbench -P ../datasets/webkb/ -c $i -E 200 -e 200 -v 0.3 -r $r -o res_nlp_$i;
    done;
done
