#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.externals import joblib
import time
import pandas as pd
import random
import sys

from utils import ddir,header

def training_sample_random(df,N = 200,class_ratio = dict(),mincount=7):
    N = int(N)
    cl = df.Categorie3
    cc = cl.groupby(cl)
    s = (cc.count() >= mincount)
    labelmaj = s[s].index
    print 'sampling ~',N,'samples for any of',len(labelmaj),'classes'
    dfs = []
    for i,cat in enumerate(labelmaj):
        if i%100==0:
            print i,'/',len(labelmaj),':'
        dfcat = df[df.Categorie3 == cat]
        sample_count = int(np.round(N*class_ratio.get(cat,1)))
        if len(dfcat)>=sample_count:
            # undersample sample_count samples : take the closest first
            rows = random.sample(dfcat.index, sample_count)
            dfs.append(dfcat.ix[rows])
        else:
            # sample all samples + oversample the remaining
            dfs.append(dfcat)
            dfcat = dfcat.iloc[np.random.randint(0, len(dfcat), size=sample_count-len(dfcat))]
            dfs.append(dfcat)
    dfsample = pd.concat(dfs)
    dfsample = dfsample.reset_index(drop=True)
    dfsample = dfsample.reindex(np.random.permutation(dfsample.index),copy=False)
    return dfsample

# NOTE : training_head is the first 1500000 rows of shuffled normalized data set used for training only
# NOTE : training_tail is the last 786885 rows of shuffled normalized data set used for validation only


##########################
# building the training set
##########################

class_ratio = joblib.load(ddir+'joblib/class_ratio')
df = pd.read_csv(ddir+'training_head.csv',sep=';',names = header()).fillna('')
for i in range(9):
    print i
    dfsample = training_sample_random(df,N=456,class_ratio=class_ratio)
    dfsample.to_csv(ddir+'training_random.csv.'+str(i),sep=';',index=False,header=False)

##########################
# building the validation set
##########################

class_ratio = joblib.load(ddir+'joblib/class_ratio')
df = pd.read_csv(ddir+'training_tail.csv',sep=';',names = header()).fillna('')
for i in range(9):
    print i
    dfsample = training_sample_random(df,N=7,class_ratio=class_ratio,mincount=1)
    dfsample.to_csv(ddir+'validation_random.csv.'+str(i),sep=';',index=False,header=False)







