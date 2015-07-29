#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

from utils import wdir,ddir,header,normalize_file,adasyn_sample,add_text
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

def vectorizer(txt):
    vec = TfidfVectorizer(
        min_df = 2,
        stop_words = ['aucune','px0'],
        max_features=234567,
        smooth_idf=True,
        norm='l2',
        sublinear_tf=True,
        use_idf=True,
        ngram_range=(1,2))
    X = vec.fit_transform(txt)
    return (vec,X)
#dftrain = pd.read_csv(ddir+'validation_normed.csv',sep=';',names = header()).fillna('')
dftrain = pd.read_csv(ddir+'training_sampled_Categorie3_200.csv',sep=';',names = header()).fillna('')
add_txt(dftrain)
dftrain = dftrain[['Categorie3','Categorie1','txt']]
Y = dftrain.Categorie3.values
(vec,X) = vectorizer(dftrain.txt)

minclass= 1000014154 # ~ 40 samples in the validation set
Xa = adasyn_sample(X,Y,minclass,5,1000) # adasynsampling
Xb = adasyn_sample(X,Y,minclass,5,10) # undersampling

l = vec.get_feature_names()
[i for i in l if '0' in i]

T = vec.inverse_transform(X)
Ta = vec.inverse_transform(Xa)
Tb = vec.inverse_transform(Xb)
