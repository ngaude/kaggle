#!/bin/sh

#sed -e '1d' training.csv | shuf > training_shuffled.csv
#sed -e '1d' test.csv| shuf > test_shuffled.csv 
##split -dl 500000 training_shuffled.csv training_shuffled_

#python 1_normalizing.py  
python 2_vectorizing.py  
python 3_neighboring.py  
python 4_training.py
python 5_predicting.py


