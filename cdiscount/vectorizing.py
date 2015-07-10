#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

import pandas as pd
import os.path
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.externals import joblib
import sys


# data & working directories

# win
#ddir = 'E:/workspace/data/cdiscount/'
#wdir = 'C:/Users/ngaude/Documents/GitHub/kaggle/cdiscount/'
# linux
ddir = '/home/ngaude/workspace/data/cdiscount/'
wdir = '/home/ngaude/workspace/github/kaggle/cdiscount/'

j_vec = ddir+'joblib/vectorizer'
j_test = ddir+'joblib/test'
j_train = ddir+'joblib/train_'

f_test = ddir+'test.csv'
f_train = ddir+'training_shuffled_'

os.chdir(wdir)

columns = [u'Identifiant_Produit', u'Categorie1', u'Categorie2', u'Categorie3', u'Description', u'Libelle', u'Marque', u'Produit_Cdiscount', u'prix']

class iterTrain():
    def __init__(self):
        
        self.df = df
    
    def __iter__(self):
        for i in range(32):
            f_train = ddir+'training_shuffled_'

        for row_index, row in self.df.iterrows():
            if row_index%10000==0:
                print row_index
            d = m = l = ''
            if type(row.Description) is str:
                d = row.Description
            if type(row.Libelle) is str:
                l = row.Libelle
            if type(row.Marque) is str:
                m = row.Marque
            txt = ' '.join([m]*3+[l]*2+[d])
            yield txt
    
    def __len__(self):
        return len(self.df)




def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

class iterText(object):
    def __init__(self, df):
        """
        Yield each document in turn, as a text.
        """
        self.df = df
    
    def __iter__(self):
        for row_index, row in self.df.iterrows():
            if row_index%10000==0:
                print row_index
            d = m = l = ''
            if type(row.Description) is str:
                d = row.Description
            if type(row.Libelle) is str:
                l = row.Libelle
            if type(row.Marque) is str:
                m = row.Marque
            txt = ' '.join([m]*3+[l]*2+[d])
            yield txt
    
    def __len__(self):
        return len(self.df)

def get_vectorizer():
    if os.path.isfile(j_vec):
        vec = joblib.load(j_vec)
        return vec
    touch(j_vec)
    # load french stop words list
    STOPWORDS = []
    with open('stop-words_french_1_fr.txt', "r") as f:
        STOPWORDS += f.read().split('\n')
    with open('stop-words_french_2_fr.txt', "r") as f:
        STOPWORDS += f.read().split('\n')
    STOPWORDS = set(STOPWORDS)
    vec = TfidfVectorizer(
        min_df = 0.00005,
        max_features=123456,
        stop_words=STOPWORDS,
        strip_accents = 'unicode',
        smooth_idf=True,
        norm='l2',
        sublinear_tf=False,
        use_idf=True,
        ngram_range=(1,3))
    df_test = pd.read_csv(f_test,sep=';')
    vec.fit(iterText(df_test))
    joblib.dump(vec, j_vec)
    return vec

# grab a vectorizer
vectorizer = get_vectorizer()

# create a pre-vectorized test text
if os.path.isfile(j_test):
    X_test = joblib.load(j_test)
else:
    touch(j_test)
    df_test = pd.read_csv(f_test,sep=';')
    X_test = vectorizer.transform(iterText(df_test))
    joblib.dump(X_test, j_test)

# create pre-vectorized train text as multiple batch of nrows

a = int(sys.argv[1])
b = int(sys.argv[2])

for i in range(a,a+b):
    print i
    f = f_train + format('%02d' % i)
    j = j_train + format('%02d' % i)
    if os.path.isfile(j):
        print j,'already exist...'
        # this file is already pending
        continue
    print 'creating'+j+'from',f,'...'
    touch(f)
    df_train = pd.read_csv(f,sep=';',header=None,names=columns)
    print 'numrows=',len(df_train)
    X_train = vectorizer.transform(iterText(df_train))
    y_train = df_train.Categorie3
    joblib.dump((X_train,y_train), j)
