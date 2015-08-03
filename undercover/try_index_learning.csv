#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

from utils import wdir,ddir,header,normalize_txt,add_txt
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from utils import itocat1,itocat2,itocat3
from utils import cat1toi,cat2toi,cat3toi
from utils import cat3tocat2,cat3tocat1,cat2tocat1
from utils import cat1count,cat2count,cat3count
from utils import training_sample
from os.path import isfile
from sklearn.metrics import f1_score

from joblib import Parallel, delayed
import time

def index_learning(dftrain,rayon,cat1,F1score=0.8):
    # select Categorie1 from rayon and training set
    ry = rayon[rayon.Categorie1 == cat1].copy()
    df = dftrain[dftrain.Categorie1 == cat1].copy()
    ry['txt'] = map(normalize_txt,ry.Categorie3_Name)
    add_txt(df)
    # vectorize Categorie3_Name as index
    vec = TfidfVectorizer(stop_words = None,min_df = 1,max_features = None,smooth_idf=True,norm='l2',sublinear_tf=False,use_idf=True,ngram_range=(1,3))
    Xr = vec.fit_transform(ry.txt)
    Xt = vec.transform(df.txt)
    # compute distance from sample to index
    D = pairwise_distances(Xt,Xr,metric='cosine')
    a = np.argmin(D,axis=1)
    df['D'] = D[range(len(a)),a]
    df['guess'] = ry.Categorie3.values[a]
    Dmin = {}
    for d in np.linspace(0,1,21):
        Yr = df[df.D<d].guess
        Yt = df[df.D<d].Categorie3
        fs = f1_score(Yt,Yr,labels=ry.Categorie3,average=None)
        for i in np.nonzero(fs > F1score)[0]:
            cat3 = ry.Categorie3.values[i]
            if cat3 in Dmin:
                continue
            Dmin[cat3] = d
    joblib.dump((vec,Dmin,F1score),ddir+'joblib/index_'+str(cat1))
    del ry,df
    return vec,Dmin,F1score

def guess_accuracy(df):
    assert 'guess' in df.columns
    assert 'Categorie3' in df.columns
    guessed = df[~df.guess.isnull()]
    total = len(guessed)
    correct = sum(guessed.guess == guessed.Categorie3)
    accuracy = 0
    if total>0:
        accuracy = float(correct)/float(total)
    return accuracy

def index_guessing(dfsample,rayon,cat1,vec,Dmin,default=None):
    if 'guess' not in dfsample.columns:
        dfsample['guess'] = None
    df = dfsample[dfsample.Categorie1 == cat1].copy()
    if len(df)==0:
        return []
    ry = rayon[rayon.Categorie1 == cat1].copy()
    add_txt(df)
    ry['txt'] = map(normalize_txt,ry.Categorie3_Name)
    Xr = vec.transform(ry.txt)
    Xt = vec.transform(df.txt)
    D = pairwise_distances(Xt,Xr,metric='cosine')
    a = np.argmin(D,axis=1)
    df['guess'] = ry.Categorie3.values[a]
    df['D'] = D[range(len(a)),a]
    return [r.guess if r.D<Dmin.get(r.guess,0) else None for i,r in df.iterrows()]

#########################
# training
#########################

dftrain= pd.read_csv(ddir+'training_shuffled_normed.csv',sep=';',names = header()).fillna('')
dfvalid = pd.read_csv(ddir+'validation_perfect.csv',sep=';',names = header()).fillna('')
rayon = pd.read_csv(ddir+'rayon.csv',sep=';').fillna('')

for cat1 in itocat1:
    vec,Dmin,__ = index_learning(dftrain,rayon,cat1,F1score=0.8)
    a = index_guessing(dfvalid,rayon,cat1,vec,Dmin)
    dfvalid.ix[dfvalid.Categorie1 == cat1, 'guess'] = a
    acc = guess_accuracy(dfvalid[dfvalid.Categorie1 == cat1])
    t = sum((dfvalid.Categorie1 == cat1))
    g = sum((dfvalid.Categorie1 == cat1) & (~dfvalid.guess.isnull()))
    print 'index',cat1,': total=',t,'guessed=',g,'accuracy=',acc 

acc = guess_accuracy(dfvalid)
t = len(dfvalid)
g = sum(~dfvalid.guess.isnull())
print 'index overall: total=',t,'guessed=',g,'accuracy=',acc 

#########################
# predicting
#########################

# add categorie1 to test based on previous resultat : we suppose stage1 to be perfect
dfresultat = pd.read_csv(ddir+'resultat44.csv',sep=';').fillna('')
dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('') 
rayon = pd.read_csv(ddir+'rayon.csv',sep=';').fillna('')
df = dftest.merge(dfresultat,'left',None,'Identifiant_Produit','Id_Produit')
df = df.merge(rayon,'left',None,'Id_Categorie','Categorie3')

for cat1 in itocat1:
    print 'index',cat1,'prediction'
    fname = ddir+'joblib/index_'+str(cat1)
    if not isfile(fname):
        print fname+' not found'
        continue
    (vec,Dmin,F1score) = joblib.load(fname)
    a = index_guessing(df,rayon,cat1,vec,Dmin)
    df.ix[df.Categorie1 == cat1, 'guess'] = a
    t = sum((df.Categorie1 == cat1))
    g = sum((df.Categorie1 == cat1) & (~df.guess.isnull()))
    print 'index',cat1,': total=',t,'guessed=',g,'F1score=',F1score 

t = len(df)
g = sum(~df.guess.isnull())
print 'index overall: total=',t,'guessed=',g 

#########################
# diffing
#########################

diff = df[(~df.guess.isnull()) & (df.guess != df.Categorie3)]
diff = diff[['Identifiant_Produit','Description','Libelle','Marque','prix','Categorie3_Name','guess']]
diff = diff.merge(rayon,'left',None,'guess','Categorie3')
diff = diff[[u'guess',u'Categorie3_Name_x',u'Categorie3_Name_y',  u'Description', u'Libelle', u'Marque', u'prix']]
diff.to_csv(ddir+'diff.csv',sep=';',index=False)
print 'finally guessing offers :',len(diff),'guesses'

#########################
# guessing
#########################

dfguess = df
filt = ((dfguess.Id_Categorie != dfguess.guess) & (~dfguess.guess.isnull()))

dfguess.ix[filt, 'Id_Categorie'] = dfguess[filt].guess
dfguess = df[['Id_Produit','Id_Categorie']].drop_duplicates()
dfguess.to_csv(ddir+'guess.csv',sep=';',index=False)

