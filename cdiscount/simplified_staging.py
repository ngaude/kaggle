#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""
from utils import wdir,ddir,header,normalize_file,iterText
from utils import MarisaTfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from os.path import basename

from utils import itocat1,itocat2,itocat3
from utils import cat1toi,cat2toi,cat3toi
from utils import cat3tocat2,cat3tocat1,cat2tocat1
from utils import cat1count,cat2count,cat3count
from utils import training_sample
from os.path import isfile
from joblib import Parallel, delayed
from multiprocessing import Manager
from sklearn.svm import LinearSVC
import time
import joblib

dfvalid = pd.read_csv(ddir+'validation_normed.csv',sep=';',names = header()).fillna('').reset_index()
dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')
dftrain = pd.read_csv(ddir+'training_sampled_Categorie3.csv',sep=';',names = header()).fillna('')

(_,vec,_) = joblib.load(ddir + 'joblib/stage1')
Xv = vec.transform(iterText(dfvalid))
Yv = dfvalid['Categorie1'].values
Zv = dfvalid['Categorie3'].values

Xt = joblib.load(ddir+'joblib/X')
Yt = dftrain['Categorie1'].values
Zt = dftrain['Categorie3'].values

#import random
#r = random.sample(range(len(df)),len(df))
#X = X[r]
#Y = Y[r]
#Z = Z[r]

dt = -time.time()

# training cl1 and dcl2
cla = LinearSVC(loss='hinge',penalty='l2')
X = Xt
Y = Yt
cla.fit(Xt,Yt)
dt += time.time()
cl1 = cla

dcl2 = {}
for cat1 in np.unique(Yt):
    f = (Yt==cat1)
    X = Xt[f]
    Z = Zt[f]
    if len(np.unique(Z))==1:
        dcl2[cat1] = Z[0]
        continue
    cla = LinearSVC(loss='hinge',penalty='l2')
    cla.fit(X,Z)
    dcl2[cat1]=cla
    # compute classifier score
    f = (Yv==cat1)
    X = Xv[f]
    Z = Zv[f]
    print 'classifier',cat1,'=',cla.score(X,Z)

# predicting based on stack SVC models

Yp = cl1.predict(Xv)
print 'score cat1 = ',sum(Yp == Yv)*1.0/len(Yv)
Zp = np.zeros(len(Yp))
for cat1 in np.unique(Yp):
    cla = dcl2[cat1]
    f = (Yp==cat1)
    X = Xv[f]
    if type(cla) is np.int64:
        Zp[f] = cla
    else:
        Zp[f] = cla.predict(X)

scv = sum(Zp == Zv)*1.0/len(Zp)
print 'score cat3 = ',scv

from scipy.sparse import vstack

# score cat1 =  0.818410183329
# score cat3 =  0.601531363977
