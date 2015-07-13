#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

import os.path
import os
import sys
import string
import Stemmer
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

class iterDataset():
    def __init__(self,fname,count=None,offset=0,y=False,columns=[u'Identifiant_Produit', u'Categorie1', u'Categorie2', u'Categorie3', u'Description', u'Libelle', u'Marque', u'Produit_Cdiscount', u'prix']):
        stopwords = []
        with open(wdir+'stop-words_french_1_fr.txt', "r") as f:
            stopwords += f.read().split('\n')
        with open(wdir+'stop-words_french_2_fr.txt', "r") as f:
            stopwords += f.read().split('\n')
        self.stopwords = set(stopwords)
        intab = string.punctuation
        outtab = ' '*len(intab)
        self.ponctuation_tab = string.maketrans(intab, outtab)
        self.stemmer = Stemmer.Stemmer('french')
        self.count = count
        self.offset = offset
        self.fname = fname 
        self.y = y
        self.columns = columns
    def __tokenize(self,txt):
        # remove digits
        txt = ''.join([i for i in txt if not i.isdigit()])
        # lower case
        txt = txt.lower()
        # remove ponctuation
        txt = txt.translate(self.ponctuation_tab)
        # remove accents
        s1 = unicode(txt,'utf-8')
        txt = unicodedata.normalize('NFD', s1).encode('ascii', 'ignore')
        # remove french stop words
        tokens = [w for w in txt.split(' ') if (len(w)>2) and (w not in self.stopwords)]
        # french stemming
        tokens = self.stemmer.stemWords(tokens)
        return tokens
    def __iter__(self):
        if self.y == True:
            ci = self.columns.index('Categorie3')
        else:
            di = self.columns.index('Description')
            li = self.columns.index('Libelle')
            mi = self.columns.index('Marque')
        j = 0
        for i,line in enumerate(open(self.fname)):
            if i < self.offset:
                continue
            if j%10000==0:
                print self.fname,j,'/',self.count,'@',self.offset
            ls = line.split(';')
            if self.y == True:
                c = ls[ci]
                txt = c
            else:
                d = ls[di]
                l = ls[li]
                m = ls[mi]
                txt = ' '.join([m]*3+[l]*2+[d])
                txt = ' '.join(self.__tokenize(txt))
            yield txt
            j+=1
            if (self.count is not None) and (j>self.count):
                break
    def __len__(self):
        return 0

##################################
# path & file

# win
#ddir = 'E:/workspace/data/cdiscount/'
#wdir = 'C:/Users/ngaude/Documents/GitHub/kaggle/cdiscount/'
# linux
ddir = '/home/ngaude/workspace/data/cdiscount/'
wdir = '/home/ngaude/workspace/github/kaggle/cdiscount/'


f_train = ddir+'training_shuffled.csv'
f_test = ddir+'test.csv'

##################################
# CREATE VECTORIZER FROM TRAIN-SET

j_vec = ddir+'joblib/vectorizer'

if os.path.isfile(j_vec):
        vec = joblib.load(j_vec)
else:
    vec = TfidfVectorizer(
        min_df = 0.00001,
        max_features=165000,
        smooth_idf=True,
        norm='l2',
        sublinear_tf=False,
        use_idf=True,
        ngram_range=(1,3))
    vec.fit(iterDataset(f_train,400000,0))
    joblib.dump(vec,j_vec)

###################################
# VECTORIZE TEST-SET

j_test = ddir+'joblib/X_test'
X_test = vec.transform(iterDataset(f_test,offset=1,columns=['Identifiant_Produit','Description','Libelle','Marque','prix']))
joblib.dump(X_test,j_test)

###################################
# VECTORIZE TRAIN-SET

for i in range(32):
    f_train = ddir+'training_shuffled_'+format('%02d' % i)
    j_train = ddir+'joblib/XY_train_'+format('%02d' % i)
    X_train = vec.transform(iterDataset(f_train))
    y_train = np.array([int(i) for i in iterDataset(f_train,y=True)])
    joblib.dump((X_train,y_train),j_train)
    del(X_train)
    del(y_train)

