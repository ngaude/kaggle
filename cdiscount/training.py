#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

from utils import wdir,ddir,header,normalize_file
from utils import MarisaTfidfVectorizer,vectorizer,iterText
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from os.path import basename

########################
# Normalisation
########################

"""
normalize_file(ddir + 'test.csv',header(test=True))
normalize_file(ddir + 'validation.csv',header())
normalize_file(ddir + 'training_shuffled.csv',header(),nrows=2000000)
"""

def score(df,vec,cla,target):
    X = vec.transform(iterText(df))
    Y = list(df[target])
    sc = cla.score(X,Y)
    return sc

def vectorizer(df):
    # 1M max_features should fit in memory, 
    # OvA will be at max 184 classes, 
    # so we can fit coef_ =  1M*184*8B ~ 1GB in memory easily
    vec = MarisaTfidfVectorizer(
        min_df = 1,
        stop_words = None,
        max_features=1000000,
        smooth_idf=True,
        norm='l2',
        sublinear_tf=True,
        use_idf=True,
        ngram_range=(1,3))
    vec.fit(iterText(df))
    return vec

def classifier(df,vec,target):
    X = vec.transform(iterText(df))
    Y = list(df[target])
#    cla = SGDClassifier(
#        loss = 'hinge',
#        n_jobs = 3,
#        penalty='elasticnet',
#        n_iter=5,
#        random_state=42)
    cla = LogisticRegression() 
    cla.fit(X,Y)
    print 'classifier training score',cla.score(X,Y)
    return cla

########################################
# Stage 1/2/3 training 
# stage1 : 1 classifier => 52 classes
# stage2 : 52 classifiers => 536 classes
# stage3 : 536 classifiers => >5K classes
########################################

# load data

nrows = 100000
dftrain = pd.read_csv(ddir+'training_shuffled_normed.csv',sep=';',names = header(),nrows=nrows).fillna('')
dfvalid = pd.read_csv(ddir+'validation_normed.csv',sep=';',names = header()).fillna('').reset_index()

# stage 1 training

fname = ddir + 'joblib/stage1'
df = dftrain
vec = vectorizer(df)
cla = classifier(df,vec,"Categorie1")
labels = np.unique(df.Categorie1)
dfv = dfvalid
sc = score(dfv,vec,cla,"Categorie1")
print '**********************************'
print 'classifier',basename(fname),'valid score',sc
print '**********************************'
joblib.dump((labels,vec,cla),fname)

# stage 2 training

cat1 = np.unique(dftrain.Categorie1)
for i,cat in enumerate(cat1):
    fname = ddir + 'joblib/stage2_'+str(cat)
    print '-'*50
    print 'training',basename(fname),':',i,'/',len(cat1)
    df = dftrain[dftrain.Categorie1 == cat].reset_index()
    labels = np.unique(df.Categorie2)
    if len(labels)==1:
        print fname,'predict 100% ',labels[0]
        joblib.dump((labels,None,None),fname)
        continue
    vec = vectorizer(df)
    cla = classifier(df,vec,"Categorie2")
    dfv = dfvalid[dfvalid.Categorie1 == cat].reset_index()
    if len(dfv)==0:
        print 'classifier',basename(fname),'cannot be validated...'
        continue
    sc = score(dfv,vec,cla,"Categorie2")
    print '*'*50
    print 'classifier',basename(fname),'valid score',sc
    print '*'*50
    joblib.dump((labels,vec,cla),fname)

# stage 3 training

cat2 = np.unique(dftrain.Categorie2)
for i,cat in enumerate(cat2):
    fname = ddir + 'joblib/stage3_'+str(cat)
    print '-'*50
    print 'training',basename(fname),':',i,'/',len(cat2)
    df = dftrain[dftrain.Categorie2 == cat].reset_index()
    labels = np.unique(df.Categorie3)
    if len(labels)==1:
        print fname,'predict 100% ',labels[0]
        joblib.dump((labels,None,None),fname)
        continue
    vec = vectorizer(df)
    cla = classifier(df,vec,"Categorie3")
    dfv = dfvalid[dfvalid.Categorie1 == cat].reset_index()
    if len(dfv)==0:
        print 'classifier',basename(fname),'cannot be validated...'
        continue
    sc = score(dfv,vec,cla,"Categorie3")
    print '*'*50
    print 'classifier',basename(fname),'valid score',sc
    print '*'*50
    joblib.dump((labels,vec,cla),fname)

