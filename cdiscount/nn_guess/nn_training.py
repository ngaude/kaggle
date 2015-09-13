#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.externals import joblib
from sklearn.neighbors import NearestNeighbors
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from os.path import basename
import random
from sklearn.externals import joblib
from joblib import Parallel, delayed
from utils import header,add_txt
from utils import itocat1,itocat2,itocat3
from utils import cat1toi,cat2toi,cat3toi
from utils import cat3tocat2,cat3tocat1,cat2tocat1
from utils import cat1count,cat2count,cat3count
from utils import MarisaTfidfVectorizer
from utils import adasyn_sample
from sklearn.preprocessing import normalize
from scipy import sparse


def vectorizer(txt):
    vec = TfidfVectorizer(
        min_df = 1,
        stop_words = None,
        smooth_idf=True,
        norm='l2',
        sublinear_tf=True,
        use_idf=True,
        ngram_range=(1,2))
    vec.fit(txt)
    return vec

def training_sample_adasyn(df,vec,N = 200,mincount=7):
    X = vec.transform(df.txt)
    Y = df.Categorie3.values
    Xt = []
    Yt = []
    for i,cat in enumerate(np.unique(Y)):
        print 'adasyn :',i
        Xt.append(adasyn_sample(X,Y,cat,K=5,n=N))
        Yt.append([cat,]*Xt[-1].shape[0])
    Xt = np.vstack(Xt) 
    Yt = np.concatenate(Yt)
    shuffle = np.random.permutation(len(Yt))
    Xt = Xt[shuffle,:]
    Yt = Yt[shuffle]
    return Xt,Yt

ddir = '/home/ngaude/workspace/data/cdiscount.auto/'


##################################################
# tfidf vectorize with adasyn balanced sampling  #
##################################################


ext = '.0' # default value

print '-'*50

import sys
if len(sys.argv)==2:
    ext = '.'+str(int(sys.argv[1]))
    print 'training nearest neighbors '+ext


dftrain = pd.read_csv(ddir+'training_sample.csv'+ext,sep=';',names = header()).fillna('')
dfvalid = pd.read_csv(ddir+'validation_sample.csv'+ext,sep=';',names = header()).fillna('')
dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')

a = set(dftrain.Identifiant_Produit)
b = set(dfvalid.Identifiant_Produit)
c = b.difference(a) 
dftrain = dftrain.drop_duplicates()
dfvalid = dfvalid.drop_duplicates()
dfvalid = dfvalid[dfvalid.Identifiant_Produit.isin(c)]
print '-'*50
print ext
print 'validation on ',len(dfvalid),'unseen samples'

add_txt(dftrain)
add_txt(dfvalid)
add_txt(dftest)

vec = vectorizer(pd.concat([dftrain.txt,dfvalid.txt,dftest.txt]))

#Xtrain,Ytrain = training_sample_adasyn(dftrain,vec,N=123,mincount=57)
#Xvalid,Yvalid = training_sample_adasyn(dfvalid,vec,N=7,mincount=1)
Xtrain = vec.transform(dftrain.txt)
Xvalid = vec.transform(dfvalid.txt)
Ytrain = dftrain.Categorie3.values
Yvalid = dfvalid.Categorie3.values

Xtest = vec.transform(dftest.txt)

##################
# FIND NEAREST NEIGHBORS
##################

train_count = Xtrain.shape[0]
valid_count = Xvalid.shape[0]
test_count = Xtest.shape[0]

nn_test = [(1,0)]*(test_count)
nn_valid = [(1,0)]*(valid_count)

def nn_median(nn):
    return np.median([ tup[0] if tup else 1 for tup in nn])

batch_size = 10000
start_time = time.time()
for i in range(0,train_count,batch_size):
    if (i/batch_size)%1==0:
        print 'neighbor:',i,'/',train_count,'median distance=',nn_median(nn_test),'time=',float(time.time() - start_time),'s'
    X = Xtrain[i:i+min(batch_size,train_count-i)]
    knn = NearestNeighbors(n_neighbors=1, algorithm='brute',metric='cosine').fit(X)
    dist,indx = knn.kneighbors(Xtest)
    nn_test = [ (dist[j,0],indx[j,0]+i) if dist[j,0]<tup[0] else tup for j,tup in enumerate(nn_test) ]
    dist,indx = knn.kneighbors(Xvalid)
    nn_valid = [ (dist[j,0],indx[j,0]+i) if dist[j,0]<tup[0] else tup for j,tup in enumerate(nn_valid) ]

##########################
# SCORE VALIDATION SET
##########################

Yvalid_pred  = map(lambda i: Ytrain[i], zip(*nn_valid)[1])
Ytest_pred  = map(lambda i: Ytrain[i], zip(*nn_test)[1])

print 'validation score',np.mean(Yvalid_pred == Yvalid)

dftest['Id_Categorie'] = Ytest_pred
dftest['Id_Produit'] = dftest.Identifiant_Produit

resultat = dftest[['Id_Produit','Id_Categorie']]
resultat.to_csv(ddir+'resultat.nn'+ext+'.csv',sep=';',index=False)

