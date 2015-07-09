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
from sklearn.neighbors import NearestNeighbors
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
j_train = ddir+'joblib/train_best'

f_test = ddir+'test.csv'
f_train = ddir+'training_best.csv'

columns = [u'Identifiant_Produit', u'Categorie1', u'Categorie2', u'Categorie3', u'Description', u'Libelle', u'Marque', u'Produit_Cdiscount', u'prix']

os.chdir(wdir)

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

# create pre-vectorized train best text
if os.path.isfile(j_train):
    (X_train,y_train) = joblib.load(j_train)
else:
    touch(j_train)
    df_train = pd.read_csv(f_train,sep=';',names = columns)
    X_train = vectorizer.transform(iterText(df_train))
    y_train = df_train.Categorie3
    joblib.dump((X_train,y_train), j_train)


def categorie_freq(df):
    # compute Categorie3 classe frequency
    sfreq = len(df)*1.
    g = df.groupby('Categorie3').Libelle.count()/sfreq
    return dict(g)

w_train = categorie_freq(df_train)

# train a SGD classifier on the X_sample very fitted sample of training according X_train distances

def train_neighborhood_sanity_check(X_train,X_test):
    n = X_test.shape[0]
    m = X_train.shape[0]
    dist = np.zeros(m)
    for off_c in range(0,m,10000):
        print off_c,'/',m
        size_c = min(off_c+10000,m) - off_c
        X_c = X_train[off_c:off_c+size_c]
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute',metric='cosine').fit(X_test)
        t_dist,_ = nbrs.kneighbors(X_c)
        print t_dist.shape
        dist[off_c:off_c+size_c] = t_dist[:,0]
    return dist

aa = train_neighborhood_sanity_check(X_train,X_test)
import matplotlib.pyplot as plt
plt.hist(aa,bins=200)
plt.show(block=False)
## shall be perky gaussian on the left side, centered around ~ 0.3 



classifier = SGDClassifier()
classifier.fit(X_train,y_train,class_weight = w_train)
print classifier.score(X_train,y_train)

########################
## RESULTAT SUBMISSION #
########################

def compare_resultat(f1,f2):
    df1 = pd.read_csv(f1,sep=';')
    df2 = pd.read_csv(f1,sep=';')
    cmp_score = sum(df1.Id_Categorie == df2.Id_Categorie)*1./len(df1)
    return cmp_score


submit_file = ddir+'resultat6.csv'
df_test = pd.read_csv(f_test,sep=';')
df_test['Id_Produit']=df_test['Identifiant_Produit']
df_test['Id_Categorie'] = classifier.predict(X_test)
df_test = df_test[['Id_Produit','Id_Categorie']]
df_test.to_csv(submit_file,sep=';',index=False)

## comparison with :
## resultat1.csv scored 15,87875%
## resultat2.csv scored 20,66930%
## resultat3.csv scored 37,52794% (train2test median distance 0.481 and sample size ~39K)
## resultat4.csv scored 43,80265% (train2test median distance 0.418 and sample size 47242)
## resultat5.csv scored 49,27511% (train2test median distance 0.361 and sample size 56812)

# modelisation of score resultat6.csv :
# based on sample = 15,3M 
# estimated distance = 0.331 = 0.481-math.log(15.3,3)*(0.481-0.418)*0.96
# estimated score = (1-0.331)*(0.3752794/(1-0.481)) + 1.5*math.log(15.3,3)/100
# estimated score = 0.5209866340683579

# we will see ...


