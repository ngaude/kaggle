#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

import numpy as np
from sklearn.externals import joblib
from sklearn.neighbors import NearestNeighbors
import time

# win
#ddir = 'E:/workspace/data/cdiscount/'
#wdir = 'C:/Users/ngaude/Documents/GitHub/kaggle/cdiscount/'
# linux
ddir = '/home/ngaude/workspace/data/cdiscount/'
wdir = '/home/ngaude/workspace/github/kaggle/cdiscount/'

(itocat1,cat1toi,itocat2,cat2toi,itocat3,cat3toi) = joblib.load(ddir+'joblib/itocat')

(Xtrain,Ytrain) = joblib.load(ddir+'joblib/XYtrain')

Xtest = joblib.load(ddir+'joblib/Xtest')

test_count = Xtest.shape[0]
train_count = Xtrain.shape[0]

neighbors = [[] for i in range(test_count)]

def neighbor_select(test_id,dist,indx):
    if len(neighbors[test_id])>100:
        neighbors[test_id].sort()
        neighbors[test_id] = neighbors[test_id][:50]
    neighbors[test_id]+= zip(dist,indx)

def neighbor_distance(k):
    return np.median([ zip(*tup[:k])[0] if tup else [1]*k for tup in neighbors])

batch_size = 1000
k = 5
start_time = time.time()
for i in range(0,train_count,batch_size):
    if (i/batch_size)%10==0:
        print 'neighbor:',i,'/',train_count,'median distance=',neighbor_distance(k),'time=',int(time.time() - start_time),'s'
    Xb = Xtrain[i:i+min(batch_size,train_count-i)]
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute',metric='cosine').fit(Xb)
    dist,indx = nbrs.kneighbors(Xtest)
    for j in range(0,test_count):
        neighbor_select(j,dist[j,:],indx[j,:]+i)

Ineighbor = np.zeros(shape=(test_count,50),dtype = int)
Dneighbor = np.zeros(shape=(test_count,50),dtype = float)

for i in range(test_count):
    neighbors[i].sort()
    Dneighbor[i,:] = zip(*neighbors[i])[0][:50]
    Ineighbor[i,:] = zip(*neighbors[i])[1][:50]

#save raw list of the top 50 at least neighbors
joblib.dump((Dneighbor,Ineighbor),ddir+'joblib/DIneighbor')

# select for each test the 5-closest neighbors
neighbors_indices = sorted(set(Ineighbor[:,:k].flatten()))

# save neighbors
Yneighbor = Ytrain[neighbors_indices]
del Ytrain
Xneighbor = Xtrain[neighbors_indices]
del Xtrain
joblib.dump((Xneighbor,Yneighbor),ddir+'joblib/XYneighbor')

