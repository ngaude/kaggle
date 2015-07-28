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

def vectorizer1(txt):
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

def vectorizer2(txt):
    vec = TfidfVectorizer(
        min_df = 2,
        stop_words = None,
        max_features=123456,
        smooth_idf=True,
        norm='l2',
        sublinear_tf=False,
        use_idf=True,
        ngram_range=(1,2))
    X = vec.fit_transform(txt)
    return (vec,X)

def vectorizer3(txt):
    vec = TfidfVectorizer(
        min_df = 2,
        stop_words = None,
        max_features=123456,
        smooth_idf=True,
        norm='l2',
        sublinear_tf=True,
        use_idf=True,
        ngram_range=(1,2))
    X = vec.fit_transform(txt)
    return (vec,X)

def vectorizer(txt):
    vec = TfidfVectorizer(
        min_df = 2,
        stop_words = None,
        max_features=123456,
        smooth_idf=True,
        norm='l2',
        sublinear_tf=True,
        use_idf=True,
        ngram_range=(1,2))
    X = vec.fit_transform(txt)
    return (vec,X)

#####################
# create sample set 
# from training set
#####################

df = pd.read_csv(ddir+'training_shuffled_normed.csv',sep=';',names = header()).fillna('').reset_index(drop=True)
#create_sample(df,'Categorie3',1000,50) # "training_sampled_Categorie3_1000.csv"
create_sample(df,'Categorie3',200,10) # "training_sampled_Categorie3_200.csv"
del df

#####################
# vectorize sample set
#####################

dfvalid = pd.read_csv(ddir+'validation_normed.csv',sep=';',names = header()).fillna('').reset_index()
dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')
dftrain = pd.read_csv(ddir+'training_sampled_Categorie3_200.csv',sep=';',names = header()).fillna('')

add_txt(dftrain)
add_txt(dfvalid)
add_txt(dftest)

if True:
    (vec,X) = vectorizer(dftrain.txt)
    Y = dftrain['Categorie1'].values
    Z = dftrain['Categorie3'].values
    joblib.dump((vec,X,Y,Z),ddir+'joblib/vecXYZ')
else:
    (vec,X,Y,Z) = joblib.load(ddir+'joblib/vecXYZ')

Xv = vec.transform(dfvalid.txt)
Yv = dfvalid['Categorie1'].values
Zv = dfvalid['Categorie3'].values

Xt = vec.transform(dftest.txt)

# training classifier for categorie1
dt = -time.time()
cla1 = LogisticRegression(C=5)
cla1.fit(X,Y)
dt += time.time()
print 'training time',dt
sct = cla1.score(X[:30000],Y[:30000])
scv = cla1.score(Xv,Yv)
print '**********************************'
print 'classifier Categorie1 training score',sct
print 'classifier Categorie3 validation score',scv
print '**********************************'
joblib.dump((vec,cla1),ddir+'joblib/stage1')

# with create_sample(df,'Categorie3',200,10)
# vectorizer1 : cat1 sct = 0.908466666667 scv = 0.811284751576 time = 241.424009085
# vectorizer2 : cat1 sct = 0.9432 scv = 0.848916692834 time = 348.451503038
# vectorizer3 : cat1 sct = 0.946233333333 scv = 0.859399531412 time = 338.183251143
# vectorizer3 + logit C=2 : cat1 sct = 0.9584 scv = 0.869133595807 time = 451.135772943
# vectorizer3 + logit C=5 : cat1 sct = 0.9722 scv = 0.874882249221 time = 480.174947977
# 
# weak regularization... survives validation ....
