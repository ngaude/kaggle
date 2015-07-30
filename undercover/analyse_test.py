#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils import ddir,header,add_txt,MarisaTfidfVectorizer,adasyn_sample
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from os.path import basename
import time
import random

from utils import itocat1,itocat2,itocat3
from utils import cat1toi,cat2toi,cat3toi
from utils import cat3tocat2,cat3tocat1,cat2tocat1
from utils import cat1count,cat2count,cat3count

def vectorizer(txt):
    vec = MarisaTfidfVectorizer(
        min_df = 2,
        stop_words = None,
        max_features=234567,
        smooth_idf=True,
        norm='l2',
        sublinear_tf=True,
        use_idf=True,
        ngram_range=(1,2))
    X = vec.fit_transform(txt)
    return (vec,X)

df36 = pd.read_csv(ddir+'resultat36.csv',sep=';')
df35 = pd.read_csv(ddir+'resultat35.csv',sep=';')
df29 = pd.read_csv(ddir+'resultat29.csv',sep=';')

df = df29
rayon = pd.read_csv(ddir+'rayon.csv',sep=';')
test = pd.read_csv(ddir+'test.csv',sep=';')

df['Categorie3'] = df.Id_Categorie.values
df['Identifiant_Produit'] = df.Id_Produit.values
df = df.merge(rayon,how='left',on='Categorie3')
df = df.merge(test,how='left',on='Identifiant_Produit')
df = df[['Id_Produit','Categorie1','Categorie2','Categorie3','Description','Libelle','Marque','prix']]
df.to_csv(ddir+'analyse_test.tsv',sep=';',index=False)

from collections import Counter
import matplotlib.pyplot as plt

f3 = df.groupby('Categorie3').Id_Produit.count()
cf3 = Counter(f3)
k3 = sorted(cf3.keys())
sum([cf3[k]*k for k in k3]) # samples count
sum([cf3[k] for k in k3]) # classes count
plt.plot(np.log([cf3[k] for k in k3]))
plt.show()

f3 = np.array(sorted(df3.keys()))
plt.plot(f3)






print 'loading...'
dftrain = pd.read_csv(ddir+'training_shuffled_normed.csv',sep=';',names = header()).fillna('')
print 'txting...'
add_txt(dftrain)

dfsample = get_sample(dftrain)
dfsample = dfsample[['Identifiant_Produit','Categorie3','Categorie1','txt']]

dfsample.to_csv(ddir+'training_sup9.csv',sep=';',index=False,header=False)

Y = dfsample.Categorie3.values
ID = dfsample.Identifiant_Produit.values
print 'vectorizing...'
vec,X = vectorizer(dfsample.txt)
print 'dumping...'
joblib.dump((vec,ID,X,Y),ddir+'joblib/vecIDXY')

# use adasyn to get synthetic balanced dataset

Xt = []
Yt = []
for i,cat in enumerate(np.unique(Y)):
    print 'adasyn :',i
    Xt.append(adasyn_sample(X,Y,cat,K=5,n=200))
    Yt.append([cat,]*Xt[-1].shape[0])

Xt = sparse.vstack(Xt) 
assert Xt.shape[0] == len(Yt)

rows = random.sample(Xt,Xt.shape[0])

# TODO : beware...
Xt[rows]

Xt = Xt[rows]

joblib.dump((vec,Xs,Ys),ddir+'joblib/vecXtYt')

def training_stage1(X,Y,Xv,Yv):
    fname = ddir + 'joblib/stage1'
    print '-'*50
    print 'training',basename(fname)
    cla = LogisticRegression(C=5)
    cla.fit(X,Y)
    labels = np.unique(Y)
    sct = cla.score(X[:10000],Y[:10000])
    scv = cla.score(Xv,Yv)
    joblib.dump((labels,cla),fname)
    del X,Y,Xv,Yv,cla
    return sct,scv

def training_stage3(X,Y,Xv,Yv,cat,i):
    fname = ddir + 'joblib/stage3_'+str(cat)
    labels = np.unique(Y)
    if len(labels)==1:
        print fname,'predict 100% ',labels[0]
        joblib.dump((labels,None,None),fname)
        scv = -1
        sct = -1
        return (sct,scv)
    cla = LogisticRegression(C=5)
    cla.fit(X,Y)
    sct = cla.score(X[:min(10000,len(df))],Y[:min(10000,len(df))])
    if len(dfv)==0:
        scv = -1
    else:
        scv = cla.score(Xv,Yv)
    print 'Stage 3.'+str(i)+':',cat,'score',sct,scv
    joblib.dump((labels,cla),fname)
    del cla
    return (sct,scv)
