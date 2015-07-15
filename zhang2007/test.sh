#!/bin/bash

#python zhang2007_impl.py -t data/train.gold.utf8 -d data/test.gold.utf8 -m model/seg.model -i 30 -b 16
date 
pypy zhang2007_impl.py -t data/train.gold.utf8 -d data/test.gold.utf8 -m model/seg.model -i 30 -b 16
date
#python zhang2007_impl.py -t data/small.train.utf8 -d data/small.train.utf8 -m model/seg.model -i 30 -b 16
