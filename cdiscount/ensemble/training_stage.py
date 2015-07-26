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
from utils import touch
import os
import sys
from joblib import Parallel, delayed
from multiprocessing import Manager

def score(df,vec,cla,target):
    X = vec.transform(iterText(df))
    Y = list(df[target])
    sc = cla.score(X,Y)
    return sc

def vectorizer_classifier(df,target):
    vec = TfidfVectorizer(
        min_df = 0.00009,
        stop_words = None,
        max_features=123456,
        smooth_idf=True,
        norm='l2',
        sublinear_tf=False,
        use_idf=True,
        ngram_range=(1,3))
    X = vec.fit_transform(iterText(df))
    Y = list(df[target])
    cla = LogisticRegression() 
    cla.fit(X,Y)
    return (vec,cla)

#######################
#######################
#######################
#######################
# training
# staged : Categorie1 =>  Categorie3
#######################
#######################
#######################
#######################


def training_stage1(dftrain,dfvalid,ensemble):
    fname = ddir + 'joblib/stage1.'+str(ensemble)
    df = dftrain
    (vec,cla) = vectorizer_classifier(df,"Categorie1")
    labels = np.unique(df.Categorie1)
    dfv = dfvalid
    sct = score(df[:30000],vec,cla,'Categorie1')
    scv = score(dfv,vec,cla,'Categorie1')
    print '**********************************'
    print 'classifier[',ensemble,'] training score',sct
    print 'classifier[',ensemble,'] validation score',scv
    print '**********************************'
    joblib.dump((labels,vec,cla),fname)
    del vec,cla
    return (sct,scv)

def training_single_stage3(ctx,cat,ensemble):
    dfvalid = ctx.dfvalid
    dftrain = ctx.dftrain
    fname = ddir + 'joblib/stage3_'+str(cat)+'.'+str(ensemble)
    print '-'*50
    print 'training',basename(fname),':',cat
    df = dftrain[dftrain.Categorie1 == cat].reset_index()
    labels = np.unique(df.Categorie3)
    if len(labels)==1:
        print fname,'predict 100% ',labels[0]
        joblib.dump((labels,None,None),fname)
        scv = -1
        sct = -1
        return (sct,scv)
    (vec,cla) = vectorizer_classifier(df,"Categorie3")
    dfv = dfvalid[dfvalid.Categorie1 == cat].reset_index()
    sct = score(df[:30000],vec,cla,'Categorie3')
    if len(dfv)==0:
        scv = -1
    else:
        scv = score(dfv,vec,cla,'Categorie3')
    print '**********************************'
    print 'classifier',cat,'training score',sct
    print 'classifier',cat,'validation score',scv
    print '**********************************'
    joblib.dump((labels,vec,cla),fname)
    del vec,cla
    return (sct,scv)

def training_stage3(dftrain,dfvalid,ensemble):
    cat1 = np.unique(dftrain.Categorie1)
    mgr = Manager()
    ctx = mgr.Namespace()
    ctx.dftrain = dftrain
    ctx.dfvalid = dfvalid
    scs = Parallel(n_jobs=3)(delayed(training_single_stage3)(ctx,cat,ensemble) for cat in cat1)
    (scts,scvs) = zip(*scs)
    return (cat1,scts,scvs)

def training(dfvalid,ensemble):
    print '>>> training :',ensemble
    fname = ddir + 'joblib/stage1.'+str(ensemble)
    ftmp =  fname+'.tmp'
    if os.path.isfile(fname) or os.path.isfile(ftmp):
        print 'already there' 
        return
    touch(ftmp)
    fname = ddir+'training_sampled_Categorie3.csv.'+str(ensemble)
    print '--- load ',fname
    dftrain = pd.read_csv(ddir+'training_sampled_Categorie3.csv.'+str(ensemble),sep=';',names = header()).fillna('').reset_index()
    print '--- train stage1'
    training_stage1(dftrain,dfvalid,ensemble)
    print '--- train stage3'
    scs = training_stage3(dftrain,dfvalid,ensemble)
    joblib.dump(scs,ddir+'joblib/scores.'+str(ensemble))
    os.remove(ftmp)
    print '<<< training :',ensemble
    del dftrain
    return

dfvalid = pd.read_csv(ddir+'validation_normed.csv',sep=';',names = header()).fillna('').reset_index()

for ensemble in range(10):
    training(dfvalid,ensemble)
     
