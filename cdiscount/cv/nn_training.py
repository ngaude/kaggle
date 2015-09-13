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
from collections import Counter

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

ddir = '/home/ngaude/workspace/data/cdiscount/'


##################################################
# tfidf vectorize with adasyn balanced sampling  #
##################################################


ext = '.cv' 

print 'training nearest neighbors '+ext


dftrain = pd.read_csv(ddir+'sample_cv.csv',sep=';').fillna('')
dfvalid = pd.read_csv(ddir+'valid_cv.csv',sep=';').fillna('')
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

def neighbor_select(ngh,test_id,dist,indx,nsize):
    if len(ngh[test_id])>nsize*2:
        ngh[test_id].sort()
        prev =  ngh[test_id]
        ngh[test_id] = list(ngh[test_id][:nsize])
        del prev
    ngh[test_id]+= zip(dist,indx)

def neighbor_median(ngh,k):
    return np.median([ zip(*tup[:k])[0] if tup else [1]*k for tup in ngh])

def neighbor_distance(X,Xnn,offset=0,neighbor_size=200):
    n = X.shape[0]
    lneighbors = [[] for i in range(Xnn.shape[0])]
    batch_size = 10000
    k = 30
    start_time = time.time()
    for i in range(0,n,batch_size):
        if (i/batch_size)%1==0:
            print 'neighbor:',offset+i,'/',n,'median distance=',neighbor_median(lneighbors,k),'time=',(time.time() - start_time),'s'
        Xbatch = X[i:i+min(batch_size,n-i)]
        knn = NearestNeighbors(n_neighbors=k, algorithm='brute',metric='cosine').fit(Xbatch)
        dist,indx = knn.kneighbors(Xnn)
        for j in range(0,Xnn.shape[0]):
            neighbor_select(lneighbors,j,dist[j,:],indx[j,:]+i+offset,neighbor_size)
        del Xbatch,knn
    return lneighbors


K = 5
IDtrain = dftrain.Categorie3.values.copy()

nnvalid= neighbor_distance(Xtrain,Xvalid,offset=0,neighbor_size=K)
IDnnvalid = np.zeros(shape=(valid_count,K),dtype = int)
Dnnvalid = np.zeros(shape=(valid_count,K),dtype = float)
for i in range(valid_count):
    nnvalid[i].sort()
    Dnnvalid[i,:] = zip(*nnvalid[i])[0][:K]
    IDnnvalid[i,:] = [IDtrain[a] for a in zip(*nnvalid[i])[1][:K]]

nntest = neighbor_distance(Xtrain,Xtest,offset=0,neighbor_size=K)
IDnntest = np.zeros(shape=(test_count,K),dtype = int)
Dnntest = np.zeros(shape=(test_count,K),dtype = float)
for i in range(test_count):
    nntest[i].sort()
    Dnntest[i,:] = zip(*nntest[i])[0][:K]
    IDnntest[i,:] = [IDtrain[a] for a in zip(*nntest[i])[1][:K]]

def vote_and_confidence(IDnn):
    Id_Categorie_list = []
    confidence_list = []
    for i in range(IDnn.shape[0]):
        freq = [(vote,cat) for cat,vote in Counter(IDnn[i,:]).iteritems()]
        freq = sorted(freq,reverse=True)
        first_categorie = freq[0][1]
        first_confidence = float(freq[0][0])/K
        Id_Categorie_list.append(first_categorie)
        confidence_list.append(first_confidence)
    return (Id_Categorie_list,confidence_list)

dfvalid['D'] = np.median(Dnnvalid,axis=1)
(cat,conf) = vote_and_confidence(IDnnvalid)
dfvalid['Categorie3_nn'] = cat
dfvalid['confidence'] = conf

dftest['D'] = np.median(Dnntest,axis=1)
(cat,conf) = vote_and_confidence(IDnntest)
dftest['Categorie3_nn'] = cat
dftest['confidence'] = conf

print 'validation score',np.mean(dfvalid.Categorie3_nn == dfvalid.Categorie3)

def save_confidence(df,mid='.test'):
    submit_file = ddir+'confidence.nn'+mid+ext+'.csv'
    df['Id_Produit']=df['Identifiant_Produit']
    df['Id_Categorie'] = df.Categorie3_nn
    df= df[['Id_Produit','Id_Categorie','confidence','D']]
    df.to_csv(submit_file,sep=';',index=False)

save_confidence(dftest,'.test')
save_confidence(dfvalid,'.valid')

print '-'*50
