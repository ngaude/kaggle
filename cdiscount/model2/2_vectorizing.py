#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

from MarisaTfidfVectorizer import MarisaTfidfVectorizer
import os
import time
import numpy as np
from sklearn.externals import joblib


#ddir = 'E:/workspace/data/cdiscount/'
#wdir = 'C:/Users/ngaude/Documents/GitHub/kaggle/cdiscount/'
ddir = '/home/ngaude/workspace/data/cdiscount/'
wdir = '/home/ngaude/workspace/github/kaggle/cdiscount/'

columns = ['Identifiant_Produit','Categorie1','Categorie2','Categorie3','Description','Libelle','Marque','Produit_Cdiscount','prix']
columns = {k:v for v,k in enumerate(columns)}


class iterText():
    def __init__(self,fname,header,nrows=None):
        self.nrows = nrows
        self.fname = fname
        self.columns = {k:v for v,k in enumerate(header.split(';'))}
    def __iter__(self):
        start_time = time.time()
        for i,line in enumerate(open(self.fname)):
            if i%10000==0:
                print self.fname,': lines=',i,'time=',int(time.time() - start_time),'s'
            if (self.nrows is not None) and (i>=self.nrows):
                break
            ls = line.split(';')
            txt = ''
            if len(ls[self.columns['Marque']])>0:
                txt += (ls[self.columns['Marque']]+' ')*5
            txt += ls[self.columns['Libelle']]+' '
            txt += ls[self.columns['Description']]
            yield txt

class iterText_old():
    def __init__(self,fname,header,nrows=None):
        self.nrows = nrows
        self.fname = fname
        self.columns = {k:v for v,k in enumerate(header.split(';'))}
    def __iter__(self):
        start_time = time.time()
        for i,line in enumerate(open(self.fname)):
            if i%10000==0:
                print self.fname,': lines=',i,'time=',int(time.time() - start_time),'s'
            if (self.nrows is not None) and (i>=self.nrows):
                break
            ls = line.split(';')
            txt = ''
            txt += (ls[self.columns['Marque']]+' ')*3
            txt += (ls[self.columns['Libelle']])*2+' '
            txt += ls[self.columns['Description']]
            yield txt

class iterY():
    def __init__(self,fname,header,nrows=None):
        self.nrows = nrows
        self.fname = fname
        self.columns = {k:v for v,k in enumerate(header.split(';'))}
    def __iter__(self):
        start_time = time.time()
        for i,line in enumerate(open(self.fname)):
            if i%10000==0:
                print self.fname,': lines=',i,'time=',int(time.time() - start_time),'s'
            if (self.nrows is not None) and (i>=self.nrows):
                break
            ls = line.split(';')
            cat1 = int(ls[self.columns['Categorie1']])
            cat2 = int(ls[self.columns['Categorie2']])
            cat3 = int(ls[self.columns['Categorie3']])
            yield (cat1,cat2,cat3) 

def get_vectorizer_old(f_x,header_x,nrows=None,max_features=100000):
    STOPWORDS = []
    with open('stop-words_french_1_fr.txt', "r") as f:
        STOPWORDS += f.read().split('\n')
    with open('stop-words_french_2_fr.txt', "r") as f:
        STOPWORDS += f.read().split('\n')
    STOPWORDS = set(STOPWORDS)
    vname = ddir+'joblib/marisa_vectorizer'
    vec = MarisaTfidfVectorizer(
        min_df = 1,
        max_features=123456,
        stop_words=STOPWORDS,
        strip_accents = 'unicode',
        smooth_idf=True,
        norm='l2',
        sublinear_tf=False,
        use_idf=True,
        ngram_range=(1,3))
    vec.fit(iterText_old(f_x,header_x,nrows))
    joblib.dump(vec,vname)
    return vec


def get_vectorizer(f_x,header_x,nrows=None,max_features=100000):
    vname = ddir+'joblib/marisa_vectorizer'
    vec = MarisaTfidfVectorizer(
        min_df = 1,
        max_features=max_features,
        smooth_idf=True,
        norm='l2',
        sublinear_tf=False,
        use_idf=True,
        ngram_range=(1,2))
    vec.fit(iterText(f_x,header_x,nrows))
    joblib.dump(vec,vname)
    return vec

#f_train= ddir+'training_shuffled_normed.csv'
#h_train = 'Identifiant_Produit;Categorie1;Categorie2;Categorie3;Description;Libelle;Marque;Produit_Cdiscount;prix'
#f_test = ddir+'test_shuffled_normed.csv'
#h_test = 'Identifiant_Produit;Description;Libelle;Marque;prix'
#nrows = 3000000
#vec = get_vectorizer(f_test,h_test,nrows,100000)
#
f_train= ddir+'training_shuffled.csv'
h_train = 'Identifiant_Produit;Categorie1;Categorie2;Categorie3;Description;Libelle;Marque;Produit_Cdiscount;prix'
f_test = ddir+'test_shuffled.csv'
h_test = 'Identifiant_Produit;Description;Libelle;Marque;prix'
nrows = 3000000
vec = get_vectorizer_old(f_test,h_test,nrows,100000)


Xtrain = vec.transform(iterText(f_train,h_train,nrows))
Ytrain = np.ndarray(shape = (nrows,3),dtype=int)
for i,r in enumerate(iterY(f_train,h_train,nrows)):
    Ytrain[i,:] = r
joblib.dump((Xtrain,Ytrain),ddir+'joblib/XYtrain')

Xtest= vec.transform(iterText(f_test,h_test))
joblib.dump(Xtest,ddir+'joblib/Xtest')


