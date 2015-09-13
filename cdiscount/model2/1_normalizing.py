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
import numpy as np
from sklearn.externals import joblib
import nltk
from bs4 import BeautifulSoup
import re
import pandas as pd
import unicodedata 
import time

#ddir = 'E:/workspace/data/cdiscount/'
#wdir = 'C:/Users/ngaude/Documents/GitHub/kaggle/cdiscount/'
ddir = '/home/ngaude/workspace/data/cdiscount/'
wdir = '/home/ngaude/workspace/github/kaggle/cdiscount/'

stopwords = []
with open(wdir+'stop-words_french_1_fr.txt', "r") as f:
    stopwords += f.read().split('\n')

with open(wdir+'stop-words_french_2_fr.txt', "r") as f:
    stopwords += f.read().split('\n')

stopwords += nltk.corpus.stopwords.words('french')
stopwords += ['voir', 'presentation']
stopwords = set(stopwords)
stemmer = Stemmer.Stemmer('french')

rayon = pd.read_csv(ddir+'rayon.csv',sep=';')
itocat1 = list(np.unique(rayon.Categorie1))
cat1toi = {cat1:i for i,cat1 in enumerate(itocat1)}
itocat2 = list(np.unique(rayon.Categorie2))
cat2toi = {cat2:i for i,cat2 in enumerate(itocat2)}
itocat3 = list(np.unique(rayon.Categorie3))
cat3toi = {cat3:i for i,cat3 in enumerate(itocat3)}

f_itocat = ddir+'joblib/itocat'
itocat = (itocat1,cat1toi,itocat2,cat2toi,itocat3,cat3toi)
joblib.dump(itocat,f_itocat)

def normalize_txt(txt):
    # remove html stuff
    txt = BeautifulSoup(txt,from_encoding='utf-8').get_text()
    # lower case
    txt = txt.lower()
    # special escaping character '...'
    txt = txt.replace(u'\u2026','.')
    txt = txt.replace(u'\u00a0',' ')
    # remove accent btw
    txt = unicodedata.normalize('NFD', txt).encode('ascii', 'ignore')
    #txt = unidecode(txt)
    # remove non alphanumeric char
    txt = re.sub('[^a-z_-]', ' ', txt)
    # remove french stop words
    tokens = [w for w in txt.split() if (len(w)>2) and (w not in stopwords)]
    # french stemming
    tokens = stemmer.stemWords(tokens)
    return ' '.join(tokens)

def normalize_price(price):
    if (price<0) or (price>100):
        price = 0
    return price

def normalize_file(fname,columns):
    columns = {k:v for v,k in enumerate(columns)}
    ofname = fname.split('.')[0]+'_normed.'+fname.split('.')[1]
    if os.path.isfile(ofname):
        print ofname,', already exists...'
        return
    ff = open(ofname,'w')
    start_time = time.time()
    for i,line in enumerate(open(fname)):
        di = columns['Description']
        li = columns['Libelle']
        mi = columns['Marque']
        pi = columns['prix']
        if i%1000 == 0:
            print fname,': lines=',i,'time=',int(time.time() - start_time),'s'
        ls = line.split(';')
        if 'Categorie1' in columns:
            #
            # category normalization
            c1i = columns['Categorie1']
            c2i = columns['Categorie2']
            c3i = columns['Categorie3']
            ls[c1i] = str(cat1toi[int(ls[c1i])])
            ls[c2i] = str(cat2toi[int(ls[c2i])])
            ls[c3i] = str(cat3toi[int(ls[c3i])])
        #
        # marque normalization
        txt = ls[mi]
        if txt == 'AUCUNE':
            txt = ''
        txt = re.sub('[^a-zA-Z0-9]', '_', txt).lower()
        ls[mi] = txt
        #
        # description normalization
        ls[di] = normalize_txt(ls[di])
        #
        # libelle normalization
        ls[li] = normalize_txt(ls[li])
        #
        # prix normalization
        ls[pi] = str(normalize_price(float(ls[pi].strip())))
        line = ';'.join(ls)
        ff.write(line+'\n')

print '#categorie1:',len(cat1toi)
print '#categorie2:',len(cat2toi)
print '#categorie3:',len(cat3toi)

fname = ddir + 'training_shuffled.csv'
columns = ['Identifiant_Produit','Categorie1','Categorie2','Categorie3','Description','Libelle','Marque','Produit_Cdiscount','prix']
normalize_file(fname,columns)

fname = ddir + 'test_shuffled.csv'
columns = ['Identifiant_Produit','Description','Libelle','Marque','prix']
normalize_file(fname,columns)

