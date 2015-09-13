#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

import os.path
import os
import numpy as np
from sklearn.externals import joblib
import sys

# data & working directories

# win
#ddir = 'E:/workspace/data/cdiscount/'
#wdir = 'C:/Users/ngaude/Documents/GitHub/kaggle/cdiscount/'
# linux
ddir = '/home/ngaude/workspace/data/cdiscount/'
wdir = '/home/ngaude/workspace/github/kaggle/cdiscount/'
f_train = ddir+'training_shuffled.csv'
s_train_prefix = ddir+'joblib/sample_'
b_train = ddir+'joblib/best_sample'
b_sample = ddir+'training_best.csv'

def select_sample(i,adist,aindx):
    if len(best[i])>50:
        best[i].sort()
        best[i] = best[i][:30]
    best[i].append((adist,aindx))

def process_sample(file_number):
    s_train = s_train_prefix + format('%02d' % file_number)
    assert os.path.isfile(s_train)
    print 'processing best match for',s_train
    (dist,indx) = joblib.load(s_train)
    assert dist.shape == indx.shape
    assert dist.shape[0] == 35065
    n = dist.shape[0]
    m = dist.shape[1]
    for i in range(n):
#        a = [(dist[i,j],indx[i,j]) for j in range(m) if dist[i,j]<0.66]
#        best[i] += a
        for j in range(m):
            select_sample(i,dist[i,j],indx[i,j])
    del(dist)
    del(indx)
    return

best = [[] for i in range(35065)]

for i in range(32):
    process_sample(i)

for i in range(35065):
    best[i].sort()

best_indx = np.zeros(shape =(35065,30),dtype = int)
best_dist = np.zeros(shape =(35065,30),dtype = float)

for i in range(35065):
    best_dist[i,:] = zip(*best[i])[0][:30]
    best_indx[i,:] = zip(*best[i])[1][:30]

joblib.dump((best_dist,best_indx),b_train)

#### reload the wonderful training set ....

(best_dist,best_indx) = joblib.load(b_train)

neighbor_c = 3

plt.hist(np.mean(best_dist[:,:neighbor_c],axis=1),bins=100)
plt.show(block=False)

very_best_indx = np.zeros(shape=(35065,neighbor_c))
very_best_indx = sorted(list(set(best_indx[:,:neighbor_c].flatten())))

# final step, extract the desired df...
j = 0
with open(b_sample,'w') as f_output:
    with open(f_train) as f_input:
        for i,l in enumerate(f_input):
            if (j>=len(very_best_indx)):
                break
            if (i%10000 == 0):
                print i,'/15786885'
            if (i == very_best_indx[j]):
                j=j+1
                f_output.write(l)



