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


dfvalid = pd.read_csv(ddir+'validation_normed.csv',sep=';',names = header()).fillna('').reset_index()
dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')
dftrain = pd.read_csv(ddir+'training_sampled_Categorie3.csv',sep=';',names = header()).fillna('')

(_,vec,_) = joblib.load(ddir + 'joblib/stage1')
Xv = vec.transform(iterText(dfvalid))
Yv = dfvalid['Categorie1'].values
Zv = dfvalid['Categorie3'].values

df = dftrain
X = joblib.load(ddir+'joblib/X')
Y = dftrain['Categorie1'].values
Z = dftrain['Categorie3'].values

#import random
#r = random.sample(range(len(df)),len(df))
#X = X[r]
#Y = Y[r]
#Z = Z[r]

import time
dt = -time.time()

from sklearn.svm import LinearSVC
cla = LinearSVC(loss='hinge',penalty='l2') # 0.8237 0.8220 (198s)
# cla = LinearSVC(multi_class = 'crammer_singer') #  0.9472 0.8226 (2434s)
# cla = LogisticRegression() # 0.9084 0.8112 (600s)
# cla = SGDClassifier(loss='modified_hubber', penalty='l2',n_jobs=2, n_iter = 5) # 0.8677 07859 (50s)
# cla = SGDClassifier(loss='log', penalty='l2',n_jobs=2, n_iter = 5, shuffle = True) # 0.6881 0.5827 (50s)
# cla = SVC(kernel='linear')
#cla.fit(X,Y)
cla.fit(X,Y)
dt += time.time()

cl1 = cla

dcl2 = {}
for cat1 in np.unique(Y):
    print cat1
    f = (Y==cat1)
    Xt = X[f]
    Yt = Y[f]
    cla = LinearSVC(loss='hinge',penalty='l2')
    cla.fit(X,Y)
    dcl2[cat1]=cla
    


