#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.externals import joblib
import time
import pandas as pd
import random
import sys

from utils import ddir,header

def training_sample_random(dftrain,N = 200,class_ratio = dict()):
    N = int(N)
    cl = dftrain.Categorie3
    cc = cl.groupby(cl)
    s = (cc.count() >= 7)
    labelmaj = s[s].index
    print 'sampling ~',N,'samples for any of',len(labelmaj),'classes'
    dfs = []
    for i,cat in enumerate(labelmaj):
        if i%100==0:
            print i,'/',len(labelmaj),':'
        df = dftrain[dftrain.Categorie3 == cat]
        sample_count = int(np.round(N*class_ratio.get(cat,1)))
        if len(df)>=sample_count:
            # undersample sample_count samples : take the closest first
            rows = random.sample(df.index, sample_count)
            dfs.append(df.ix[rows])
        else:
            # sample all samples + oversample the remaining
            dfs.append(df)
            df = df.iloc[np.random.randint(0, len(df), size=sample_count-len(df))]
            dfs.append(df)
    dfsample = pd.concat(dfs)
    dfsample = dfsample.reset_index(drop=True)
    dfsample = dfsample.reindex(np.random.permutation(dfsample.index),copy=False)
    return dfsample

# NOTE : training_normed is the first 1560000 rows of shuffled normalized training set
# NOTE : validation_normed is the last 186885 rows of shuffled normalized training set

dftrain = pd.read_csv(ddir+'training_normed.csv',sep=';',names = header(),nrows=15000000).fillna('')
class_ratio = joblib.load(ddir+'joblib/class_ratio')

dfsample = training_sample_random(dftrain,N=456,class_ratio=class_ratio)

if len(sys.argv)<2:
    ensemble = ''
else:
    ensemble = '.'+sys.argv[1]

print ddir+'training_random.csv'+ensemble

dfsample.to_csv(ddir+'training_random.csv'+ensemble,sep=';',index=False,header=False)
