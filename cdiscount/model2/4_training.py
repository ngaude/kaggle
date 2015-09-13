#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.externals import joblib
import sys

#ddir = 'E:/workspace/data/cdiscount/'
#wdir = 'C:/Users/ngaude/Documents/GitHub/kaggle/cdiscount/'
ddir = '/home/ngaude/workspace/data/cdiscount/'
wdir = '/home/ngaude/workspace/github/kaggle/cdiscount/'

f_itocat = ddir+'joblib/itocat'
(itocat1,cat1toi,itocat2,cat2toi,itocat3,cat3toi) = joblib.load(f_itocat)

(X,Y) = joblib.load(ddir+'joblib/XYneighbor')
Y=Y[:,2]
classes = np.unique(Y)

classifier = SGDClassifier(loss = 'hinge',n_jobs = 3,penalty='l2')

classifier.partial_fit(X,Y,classes = classes)

classifier.sparsify()

#
#nrows = 1000
#trainrows = Xtrain.shape[0]
#epochs = 5 * trainrows / nrows
#for i in range(epochs):
#    a = np.random.randint(trainrows,size=nrows)
#    Xi = Xtrain[a,:]
#    Yi = Ytrain[a]
#    print 'partial_fit',i,'/',epochs
#    classifier.partial_fit(Xi,Yi,classes = cat3toi.keys())
#
#print 'train',classifier.score(Xtrain[:30000],Ytrain[:30000])
#print 'test',classifier.score(Xtest,Ytest)
#
# cat1
# train 0.895866666667
# test 0.891
#
# cat2
# test 0.815433333333
# train 0.841566666667

# cat3
# train 0.756566666667
# test 0.733066666667

fname  = ddir+'joblib/classifier'
joblib.dump(classifier,fname)

