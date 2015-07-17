#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

import numpy as np
from sklearn.externals import joblib
import time
import pandas as pd
import time

# win
#ddir = 'E:/workspace/data/cdiscount/'
#wdir = 'C:/Users/ngaude/Documents/GitHub/kaggle/cdiscount/'
# linux
ddir = '/home/ngaude/workspace/data/cdiscount/'
wdir = '/home/ngaude/workspace/github/kaggle/cdiscount/'

(Xtrain,Ytrain) = joblib.load(ddir+'joblib/XYtrain')

Y = Ytrain[:,2]

(D,I) = joblib.load(ddir+'joblib/DIneighbor')

Xtest = joblib.load(ddir+'joblib/Xtest')

test_count = Xtest.shape[0]


import itertools
import operator

def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]

#import itertools
#import operator
#
#L = [('grape', 100), ('grape', 3), ('apple', 15), ('apple', 10),
#     ('apple', 4), ('banana', 3)]
#
#def accumulate(l):
#    it = itertools.groupby(l, operator.itemgetter(0))
#    for key, subiter in it:
#       yield key, sum(item[1] for item in subiter) 
#
#list(accumulate(L))
#L = [(Y[i],1-d) for i,d in zip(I[j,:],D[j,:])]

Ytest = np.zeros(test_count,dtype=int)
start_time = time.time()
for j in range(test_count):
    L = [Y[i] for i in I[j,:]]
    Ytest[j] = most_common(L)
    if j%1000==0:
        print 'majority:',j,'/',test_count,'time=',int(time.time() - start_time),'s'


cat3 = Ytest

submit_file = ddir+'resultat12.csv'
print 'writing >>>',submit_file
fname = ddir + 'test_shuffled.csv'
names = ['Identifiant_Produit','Description','Libelle','Marque','prix']
df_test = pd.read_csv(fname,sep=';',names=names)

df_test['Id_Produit']=df_test['Identifiant_Produit']
df_test['Id_Categorie'] = cat3
df_test = df_test[['Id_Produit','Id_Categorie']]

df_test.to_csv(submit_file,sep=';',index=False)
print 'writing <<<'




