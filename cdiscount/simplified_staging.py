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

def create_sample(df,label,mincount,maxsampling):
    fname = ddir+'training_sampled_'+label+'.csv'
    dfsample = training_sample(df,label,mincount,maxsampling)
    dfsample.to_csv(fname,sep=';',index=False,header=False)
    return dfsample

def add_txt(df):
    assert 'Marque' in df.columns
    assert 'Libelle' in df.columns
    assert 'Description' in df.columns
    df['txt'] = (df.Marque+' ')*3+(df.Libelle+' ')*2+df.Description
    return df

def vectorizer(txt):
    vec = TfidfVectorizer(
        min_df = 0.00009,
        stop_words = None,
        max_features=123456,
        smooth_idf=True,
        norm='l2',
        sublinear_tf=False,
        use_idf=True,
        ngram_range=(1,3))
    X = vec.fit_transform(txt)
    return (vec,X)

#####################
# create sample set 
# from training set
#####################

#df = pd.read_csv(ddir+'training_shuffled_normed.csv',sep=';',names = header()).fillna('').reset_index(drop=True)
#create_sample(df,'Categorie3',1000,50)
#del df

#####################
# vectorize sample set
#####################

dfvalid = pd.read_csv(ddir+'validation_normed.csv',sep=';',names = header()).fillna('').reset_index()
dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')
dftrain = pd.read_csv(ddir+'training_sampled_Categorie3.csv',sep=';',names = header()).fillna('')

add_txt(dftrain)
add_txt(dfvalid)
add_txt(dftest)

(vec,X) = vectorizer(dftrain.txt)
Y = dftrain['Categorie1'].values
Z = dftrain['Categorie3'].values

Xv = vec.transform(dfvalid.txt)
Yv = dfvalid['Categorie1'].values
Zv = dfvalid['Categorie3'].values

Xt = vec.transform(dftest.txt)

dt = -time.time()
# training classifier for categorie1
cla1 = LogisticRegression()
cla1.fit(X,Y)
dt += time.time()
print dt

joblib.dump((vec,cla1),ddir+'joblib/stage1')

cla3 = {}
for cat1 in np.unique(Y):
    f = (Y==cat1)
    Xs = X[f]
    Zs = Z[f]
    if len(np.unique(Zs))==1:
        cla2[cat1] = (np.unique(Zs),None)
        continue
    cla = LogisticRegression()
    cla.fit(Xs,Zs)
    cla3[cat1]=cla
    joblib.dump((vec,cla),ddir+'joblib/stage3_')
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
