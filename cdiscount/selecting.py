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


# data & working directories

# win
#ddir = 'E:/workspace/data/cdiscount/'
#wdir = 'C:/Users/ngaude/Documents/GitHub/kaggle/cdiscount/'
# linux
ddir = '/home/ngaude/workspace/data/cdiscount/'
wdir = '/home/ngaude/workspace/github/kaggle/cdiscount/'

j_vec = ddir+'joblib/vectorizer'
j_test = ddir+'joblib/test'
j_train = ddir+'joblib/train_'
s_train = ddir+'joblib/sample_'

os.chdir(wdir)

# load a pre-vectorized test text
assert os.path.isfile(j_test)
X_test = joblib.load(j_test)

# load pre-vectorized train text as multiple batch of nrows
file_number = int(sys.argv[1])
j_train = j_train + format('%02d' % file_number)
s_train = s_train + format('%02d' % file_number)
assert  os.path.isfile(j_train)
(X_train,y_train) = joblib.load(j_train)

neighbor_c = 5
size_c = 10000

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
    indx[:,neighbor_c*i:neighbor_c*(i+1)] = t_indx

joblib.dump((dist,indx),s_train)




sorting = np.argsort(dist, axis=1)

best_dist=np.zeros(shape=(n,3),dtype=float)
best_idx=np.zeros(shape=(n,3),dtype=int)

for i in range(n):
    best_dist[i,0] = dist[i,sorting[i,0]]
    best_dist[i,1] = dist[i,sorting[i,1]]
    best_dist[i,2] = dist[i,sorting[i,2]]
    best_idx[i,0] = idx[i,sorting[i,0]]
    best_idx[i,1] = idx[i,sorting[i,1]]
    best_idx[i,2] = idx[i,sorting[i,2]]

best_idx.shape = best_idx.shape[0]*best_idx.shape[1]


