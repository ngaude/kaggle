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

###################
# define ensemble #
###################

ext = '.0' # default value

import sys
if len(sys.argv)==2:
    assert int(sys.argv[1]) in range(9)
    ext = '.'+str(int(sys.argv[1]))

print '*'*50
print 'training with random <'+ext+'>'
print '*'*50

def vectorizer_stage3(txt):
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

def filter_cat1(df,cat1):
    return df[df.Categorie1 == cat1].reset_index(drop=True)

def pfit(p,X,Y,Xvs,Yvs):
    tt = -time.time()
    
    cla = LogisticRegression(C=p['C'],penalty='l2',class_weight = 'auto')
    cla.fit(X,Y)
    tt += time.time()
    scs = [cla.score(Xv,Yv) for Xv,Yv in zip(Xvs,Yvs)]
    sc = (np.mean(scs),np.std(scs))
    print tt,p,sc
    return (sc,cla)

def best_classifier(X,Y,Xvs,Yvs):
    parameters = {'C':[3,13,67,330,1636,8103]}
    pg = ParameterGrid(parameters)
    clas = Parallel(n_jobs=4)(delayed(pfit)(p,X,Y,Xvs,Yvs) for p in pg)
    clas.sort(reverse=True)
    (sc,cla) = clas[0]
    print '-'*20
    print 'best is ',cla,sc
    print '-'*20
    return cla,sc

def training_stage3(dft,dfvs,cat1,i):
    fname = ddir + 'joblib/stage3_'+str(cat1)
    dft = filter_cat1(dft,cat1)
    dfvs = [filter_cat1(dfv,cat1) for dfv in dfvs]
    labels = np.unique(dft.Categorie3)
    if len(labels)==1:
        # only one label
        vec = None
        cla = None
        scv = (-1,0)
        joblib.dump((labels,vec,cla,scv),fname+ext)
        print 'training',cat1,': skipped, there is only one label to predict'
        return scv
    vec,X = vectorizer_stage3(dft.txt)
    Y = dft['Categorie3'].values
    if sum([len(dfv) for dfv in dfvs]) == 0:
        print 'default parameter'
        cla = LogisticRegression()
        cla.fit(X,Y)
        scv = (-1,0)
    else:
        # performs a gridsearch
        Xvs = [ vec.transform(dfv.txt) for dfv in dfvs]
        Yvs = [ dfv['Categorie3'].values for dfv in dfvs]
        cla,scv = best_classifier(X,Y,Xvs,Yvs)
    print 'training',cat1,'\t\t(',i,') N=',len(dft),'K=',len(labels),': mean =',scv[0],'dev=',scv[1]
    joblib.dump((labels,vec,cla,scv),fname+ext)
    del vec,cla
    return scv

#################
# prepare train #
#################
dftrain = pd.read_csv(ddir+'training_random.csv'+ext,sep=';',names = header()).fillna('')
add_txt(dftrain)
dftrain = dftrain[['Categorie3','Categorie1','txt']]

#################
# prepare valid #
#################
dfvs = [pd.read_csv(ddir+'validation_random.csv.'+str(i),sep=';',names = header()).fillna('') for i in range(9)]
for i in range(9):
    add_txt(dfvs[i])
    dfvs[i] = dfvs[i][['Identifiant_Produit','Categorie3','Categorie1','txt']]

#################
# prepare test  #
#################

for i,cat1 in enumerate(np.unique(dftrain.Categorie1)):
    dt = -time.time()
    print '*'*50
    print '>> cv training',cat1,'for',ext,' #'+str(i)+'/50'
    sc,st = training_stage3(dftrain,dfvs,cat1,i)
    print '<< best is',sc,'with variance',st
    print '*'*50
    dt += time.time()
    print 'time=',dt

