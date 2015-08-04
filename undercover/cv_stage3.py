#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

from utils import wdir,ddir,header,normalize_file,add_txt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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
import time

from sklearn.grid_search import ParameterGrid 
from sklearn.grid_search import GridSearchCV

def vectorizer(txt):
    vec = TfidfVectorizer(
        min_df = 1,
        stop_words = None,
        smooth_idf=True,
        norm='l2',
        sublinear_tf=True,
        use_idf=True,
        ngram_range=(1,2))
    X = vec.fit_transform(txt)
    return (vec,X)

def pfit(p,X,Y,Xv,Yv):
    cla = LogisticRegression(C=p['C'],dual=p['dual'],class_weight=p['class_weight'],penalty=p['penalty'])
    cla.fit(X,Y)
    sc = cla.score(Xv,Yv)
    print p,sc
    return (cla,sc)

def best_classifier(X,Y,Xv,Yv):
    parameters = [
        {'C':[1,3,10,30,100,300,1000],'dual':[False],'class_weight':['auto',None],'penalty':['l1','l2']},
        {'C':[1,3,10,30,100,300,1000],'dual':[True],'class_weight':['auto',None],'penalty':['l2']}]
    pg = ParameterGrid(parameters)
    clas = Parallel(n_jobs=3)(delayed(pfit)(p,X,Y,Xv,Yv) for p in pg)
    clas.sort(reverse=True)
    cla,sc = clas[0]
    return cla,sc

def training_stage3(dftrain,dfvalid,cat1,i):
    fname = ddir + 'joblib/stage3_'+str(cat1)
    df = dftrain[dftrain.Categorie1 == cat1].reset_index(drop=True)
    dfv = dfvalid[dfvalid.Categorie1 == cat1].reset_index(drop=True)
    labels = np.unique(df.Categorie3)
    if len(labels)==1:
        joblib.dump((labels,None,None),fname)
        scv = -1
        sct = -1
        print 'training',cat1,'\t\t(',i,') : N=',len(df),'K=',len(labels)
        print 'training',cat1,'\t\t(',i,') : training=',sct,'validation=',scv
        return (sct,scv)
    vec,X = vectorizer(df.txt)
    Y = df['Categorie3'].values
    # FIXME : shall be at least one element in Xv
    # FIXME : shall be at least one element in Xy
    # FIXME : shall be at least one element in Xy
    # FIXME : shall be at least one element in Xy
    labels = np.unique(df.Categorie3)
    if len(dfv)==0:
        scv = -1
        cla = None
    else:
        Xv = vec.transform(dfv.txt)
        Yv = dfv['Categorie3'].values
        cla,scv = best_classifier(X,Y,Xv,Yv)
    print 'training',cat1,'\t\t(',i,') : N=',len(df),'K=',len(labels)
    print 'training',cat1,'\t\t(',i,') : validation=',scv
    joblib.dump((labels,vec,cla),fname)
    del vec,cla
    return (sct,scv)

dftrain = pd.read_csv(ddir+'training_random.csv.0',sep=';',names = header()).fillna('')
dfvalid = pd.read_csv(ddir+'validation.csv.0',sep=';',names = header()).fillna('')
dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')

add_txt(dftrain)
add_txt(dfvalid)
add_txt(dftest)

dftrain = dftrain[['Categorie3','Categorie1','txt']]
dfvalid = dfvalid[['Categorie3','Categorie1','txt']]
dftest = dftest[['Identifiant_Produit','txt']]



cat1  =1000000235

dfts = []
dfvs = []
dft = dftrain[dftrain.Categorie1 == cat1].reset_index(drop=True)
dfv = dfvalid[dfvalid.Categorie1 == cat1].reset_index(drop=True)

dt = -time.time()
sct,scv = training_stage3(dft,dfv,cat1,0)
dt += time.time()

(labels,vec,cla) = joblib.load(ddir + 'joblib/stage3_'+str(cat1))
print cla.best_params_
