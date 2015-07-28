#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

import pandas as pd
import os.path
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.externals import joblib
import sys
from sklearn.neighbors import NearestNeighbors

def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

# data & working directories

# win
#ddir = 'E:/workspace/data/cdiscount/'
#wdir = 'C:/Users/ngaude/Documents/GitHub/kaggle/cdiscount/'
# linux
ddir = '/home/ngaude/workspace/data/cdiscount/'
wdir = '/home/ngaude/workspace/github/kaggle/cdiscount/'

j_vec = ddir+'joblib/vectorizer'
j_test = ddir+'joblib/test'
j_train_prefix = ddir+'joblib/train_'
s_train_prefix = ddir+'joblib/sample_'

os.chdir(wdir)

# size of the batch
size_c = 10000
# number of selected neighbor within each batch
neighbor_c = 5

def sampling_vectorized_file(file_number):
    j_train = j_train_prefix + format('%02d' % file_number)
    s_train = s_train_prefix + format('%02d' % file_number)
    if os.path.isfile(s_train):
        print s_train,'already exist...'
        # this file is already pending
        return
    touch(s_train)
    print 'creating'+s_train+'from',j_train,'...'
    # load pre-vectorized train text as multiple batch of nrows
    assert  os.path.isfile(j_train)
    (X_train,y_train) = joblib.load(j_train)
    n = X_test.shape[0] # 35065
    m = X_train.shape[0]/size_c*neighbor_c # 500000/10000
    dist=np.zeros(shape=(n,m),dtype=float)
    indx=np.zeros(shape=(n,m),dtype=int)
    for i in range(m/neighbor_c):
        print j_train,':',i,'/',m/neighbor_c
        off_c = i*size_c
        X_c = X_train[off_c:off_c+size_c]
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute',metric='cosine').fit(X_c)
        t_dist,t_indx = nbrs.kneighbors(X_test)
        dist[:,neighbor_c*i:neighbor_c*(i+1)] = t_dist
        indx[:,neighbor_c*i:neighbor_c*(i+1)] = t_indx+off_c+file_number*500000
    joblib.dump((dist,indx),s_train)
    return dist,indx

# load a pre-vectorized test text
assert os.path.isfile(j_test)
X_test = joblib.load(j_test)

a = int(sys.argv[1])
b = int(sys.argv[2])

for i in range(a,a+b):
    sampling_vectorized_file(i)
