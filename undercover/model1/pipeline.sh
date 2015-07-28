#!/bin/sh

sed -e '1d' training.csv | shuf > training_shuffled.csv
split -dl 500000 training_shuffled.csv training_shuffled_

python vectorizing.py 0 10
python vectorizing.py 10 10
python vectorizing.py 20 31

python sampling.py 0 1
python sampling.py 1 1
python sampling.py 2 1
python sampling.py 3 1
python sampling.py 4 1
python sampling.py 5 1
python sampling.py 6 1
python sampling.py 7 1
python sampling.py 8 1
python sampling.py 9 1
python sampling.py 10 1
python sampling.py 11 1
python sampling.py 12 1
python sampling.py 13 1
python sampling.py 14 1
python sampling.py 15 1
python sampling.py 16 1
python sampling.py 17 1
python sampling.py 18 1
python sampling.py 19 1
python sampling.py 20 1
python sampling.py 21 1
python sampling.py 22 1
python sampling.py 23 1
python sampling.py 24 1
python sampling.py 25 1
python sampling.py 26 1
python sampling.py 27 1
python sampling.py 28 1
python sampling.py 29 1
python sampling.py 30 1
python sampling.py 31 1

python selecting.py

python predicting.py



