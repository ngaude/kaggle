#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

from utils import wdir,ddir,header
from utils import iterText
from utils import itocat1,itocat2,itocat3
from utils import cat1toi,cat2toi,cat3toi
from utils import cat3tocat2,cat3tocat1
from utils import cat1count,cat2count,cat3count
from os.path import isfile

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from os.path import basename


def log_proba(df,vec,cla):
    X = vec.transform(iterText(df))
    lp = cla.predict_log_proba(X)
    return lp

########################################
# Stage 1/2/3 training 
# stage1 : 1 classifier => 52 classes
# stage2 : 52 classifiers => 536 classes
# stage3 : 536 classifiers => 4789 classes
########################################

# load data

dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')
dfvalid = pd.read_csv(ddir+'validation_normed.csv',sep=';',names = header()).fillna('')

df = dfvalid
n = len(df)

stage1_log_proba = np.full(shape=(n,cat1count),-666.666,dtype = float)
stage2_log_proba = np.full(shape=(n,cat2count),-666.666,dtype = float)
stage3_log_proba = np.full(shape=(n,cat3count),-666.666,dtype = float)

# stage 1 log proba filling
fname = ddir + 'joblib/stage1'
(labels,vec,cla) = joblib.load(fname)
lp = log_proba(df,vec,cla)
for i,k in enumerate(cla.classes_):
    j = cat1toi[k]
    stage1_log_proba[:,j] = lp[:,i]


# stage 2 log proba filling
for ii,cat in enumerate(itocat1):
    fname = ddir + 'joblib/stage2_'+str(cat)
    print '-'*50
    print 'predicting',basename(fname),':',ii,'/',len(itocat1)
    if not isfile(fname): 
        continue
    (labels,vec,cla) = joblib.load(fname)
    if len(labels)==1:
        k = labels[0]
        j = cat2toi[k]
        stage2_log_proba[:,j] = 0
        continue
    lp = log_proba(df,vec,cla)
    for i,k in enumerate(cla.classes_):
        j = cat2toi[k]
        stage2_log_proba[:,j] = lp[:,i]

# TODO : stage 3 log proba
# TODO : combine linear probabilistic scores
# TODO : fingers crossed for this to rule the kaggle

