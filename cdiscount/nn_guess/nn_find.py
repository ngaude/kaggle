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

def vectorizer(txt):
    vec = MarisaTfidfVectorizer(
        min_df = 2,
        max_features = 1000000,
        stop_words = None,
        smooth_idf=True,
        norm='l2',
        sublinear_tf=True,
        use_idf=True,
        ngram_range=(1,2))
    vec.fit(txt)
    return vec

#############################################
# prepare a full text set of data for tfidf #
#############################################
# 
# test = pd.read_csv('/home/ngaude/workspace/data/cdiscount/test_normed.csv',names=header(True),sep=';').fillna('')
# tail = pd.read_csv('/home/ngaude/workspace/data/cdiscount/training_tail.csv',names=header(),sep=';').fillna('')
# head = pd.read_csv('/home/ngaude/workspace/data/cdiscount/training_head.csv',names=header(),sep=';').fillna('')
# add_txt(test)
# add_txt(tail)
# add_txt(head)
# 
# test = test[['Identifiant_Produit','txt']]
# tail = tail[['Identifiant_Produit','txt']]
# head = head[['Identifiant_Produit','txt']]
# 
# df = pd.concat([tail,test,head],copy=False)
# df.sort('Identifiant_Produit',inplace=True)
# df = df.reset_index(drop=True)
# 
# df.to_csv('/home/ngaude/workspace/data/cdiscount/fulltext.csv',sep=';',index=False)
# 

#############################################
# vectorize the full text                   #
#############################################
# 
# class iterFullText(object):
#     def __init__(self,sampling=1):
#         self.sampling = sampling
#     def __iter__(self):
#         with open('/home/ngaude/workspace/data/cdiscount/fulltext.csv', 'r') as f:
#             f.readline() # skip header
#             for line in f:
#                 tt = line.strip().split(';')
#                 assert len(tt)==2
#                 Identifiant_Produit = int(tt[0])
#                 txt = tt[1] 
#                 if Identifiant_Produit % 10000 == 0:
#                     print Identifiant_Produit
#                 if Identifiant_Produit % (self.sampling)==0:
#                     yield txt
# 
# vec = vectorizer(iterFullText(sampling=2)) # memory issue, cannot 
# X = vec.transform(iterFullText())
# 
# joblib.dump((vec,X),'/home/ngaude/workspace/data/cdiscount/joblibvecX')
# 



##################
# FIND NEAREST NEIGHBORS
##################

(vec,X) = joblib.load('/home/ngaude/workspace/data/cdiscount/joblibvecX')
# joblib.dump((vec,X),'/home/ngaude/workspace/data/cdiscount/joblib/vecX')

train_count = 15786885
test_count = 35065

Xtrain = X[:train_count,:]
Xtest = X[train_count:,:]
del X
del vec

assert Xtrain.shape[0] == train_count
assert Xtest.shape[0] == test_count

nn = [(1,0)]*35065

def nn_median(nn):
    return np.median([ tup[0] if tup else 1 for tup in nn])

batch_size = 10000
start_time = time.time()
for i in range(0,train_count,batch_size):
    if (i/batch_size)%1==0:
        print 'neighbor:',i,'/',train_count,'median distance=',nn_median(nn),'time=',float(time.time() - start_time),'s'
    X = Xtrain[i:i+min(batch_size,train_count-i)]
    knn = NearestNeighbors(n_neighbors=1, algorithm='brute',metric='cosine').fit(X)
    dist,indx = knn.kneighbors(Xtest)
    nn = [ (dist[j,0],indx[j,0]+i) if dist[j,0]<tup[0] else tup for j,tup in enumerate(nn) ]

##########################
# JOIN TEST WITH NEIGHBORS
##########################



df = pd.read_csv('test.csv',sep=';')
df['D'] = zip(*nn)[0]
df['nn'] = zip(*nn)[1]

df.to_csv('test_nn.csv',sep=';')

train = pd.read_csv('/home/ngaude/workspace/data/cdiscount/training.csv',sep=';')

dfm = df.merge(train,'left',None,'nn','Identifiant_Produit',suffixes=('_test', '_nn'))


dfm.to_csv('test_nn.csv',sep=';')
