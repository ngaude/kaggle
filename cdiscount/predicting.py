#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

from utils import wdir,ddir,header
from utils import iterText
from utils import itocat1,itocat2,itocat3
from utils import cat1toi,cat2toi,cat3toi
from utils import cat3tocat2,cat3tocat1,cat2tocat1
from utils import cat1count,cat2count,cat3count
from os.path import isfile

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from os.path import basename


def log_proba(df,vec,cla):
    X = vec.transform(iterText(df))
    lp = cla.predict_log_proba(X)
    return lp

########################################
# Stage 1/2/3 training 
# stage1 : 1 classifier => 52 classes
# stage2 : 52 classifiers => 536 classes
# stage3 : 536 classifiers => 4789 classes
########################################

# load data

dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')
dfvalid = pd.read_csv(ddir+'validation_normed.csv',sep=';',names = header()).fillna('')

df = dftest
n = len(df)

stage1_log_proba = np.full(shape=(n,cat1count),fill_value = -666.,dtype = float)

# stage 1 log proba filling
fname = ddir + 'joblib/stage1'
(labels,vec,cla) = joblib.load(fname)
lp = log_proba(df,vec,cla)
for i,k in enumerate(cla.classes_):
    j = cat1toi[k]
    stage1_log_proba[:,j] = lp[:,i]

del labels,vec,cla

stage2_log_proba = np.full(shape=(n,cat2count),fill_value = -666.,dtype = float)

# stage 2 log proba filling
for ii,cat in enumerate(itocat1):
    fname = ddir + 'joblib/stage2_'+str(cat)
    print '-'*50
    print 'predicting',basename(fname),':',ii,'/',len(itocat1)
    if not isfile(fname): 
        continue
    (labels,vec,cla) = joblib.load(fname)
    if len(labels)==1:
        k = labels[0]
        j = cat2toi[k]
        stage2_log_proba[:,j] = 0
        continue
    lp = log_proba(df,vec,cla)
    for i,k in enumerate(cla.classes_):
        j = cat2toi[k]
        stage2_log_proba[:,j] = lp[:,i]
    del labels,vec,cla

stage3_log_proba = np.full(shape=(n,cat3count),fill_value = -666.,dtype = float)

# stage 3 log proba filling
for ii,cat in enumerate(itocat2):
    fname = ddir + 'joblib/stage3_'+str(cat)
    print '-'*50
    print 'predicting',basename(fname),':',ii,'/',len(itocat2)
    if not isfile(fname): 
        continue
    (labels,vec,cla) = joblib.load(fname)
    if len(labels)==1:
        k = labels[0]
        j = cat3toi[k]
        stage3_log_proba[:,j] = 0
        continue
    lp = log_proba(df,vec,cla)
    for i,k in enumerate(cla.classes_):
        j = cat3toi[k]
        stage3_log_proba[:,j] = lp[:,i]
    del labels,vec,cla

# TODO : stage 3 log proba
# TODO : combine linear probabilistic scores
# TODO : fingers crossed for this to rule the kaggle

if (df is dfvalid):
    fname = ddir+'/joblib/log_proba_valid'
else:
    fname = ddir+'/joblib/log_proba_test'

joblib.dump((stage1_log_proba,stage2_log_proba,stage3_log_proba),fname)

##################
# (stage1_log_proba,stage2_log_proba,stage3_log_proba) = joblib.load(fname)
##################

##################
# bayes rulez ....
##################

# P(cat2) = P(cat2|cat1)*P(cat1)
################################
for i in range(stage2_log_proba.shape[1]):
    cat2 = itocat2[i]
    cat1 = cat2tocat1[cat2]
    j = cat1toi[cat1]
    stage2_log_proba[:,i] += stage1_log_proba[:,j]


# P(cat3) = P(cat3|cat2)*P(cat2) = P(cat3|cat2)*P(cat2|cat1)*P(cat1)
####################################################################
for i in range(stage3_log_proba.shape[1]):
    cat3 = itocat3[i]
    cat2 = cat3tocat2[cat3]
    j = cat2toi[cat2]
    stage3_log_proba[:,i] += stage2_log_proba[:,j]

predict_cat1 = [itocat1[i] for i in np.argmax(stage1_log_proba,axis=1)]
predict_cat2 = [itocat2[i] for i in np.argmax(stage2_log_proba,axis=1)]
predict_cat3 = [itocat3[i] for i in np.argmax(stage3_log_proba,axis=1)]

def submit():
    submit_file = ddir+'resultat.csv'
    df['Id_Produit']=df['Identifiant_Produit']
    df['Id_Categorie'] = predict_cat3
    df= df[['Id_Produit','Id_Categorie']]
    df.to_csv(submit_file,sep=';',index=False)

if df is dfvalid:
    score_cat1 = sum(dfvalid.Categorie1 == predict_cat1)*1.0/n
    score_cat2 = sum(dfvalid.Categorie2 == predict_cat2)*1.0/n
    score_cat3 = sum(dfvalid.Categorie3 == predict_cat3)*1.0/n
    print 'dfvalid scores =',score_cat1,score_cat2,score_cat3
else:
    submit(df)

#############################################################################
# resultat16.csv :
# nrows = 100K, 
# dfvalid scores = 0.681795125722 0.443322625057 0.296973503055
# dftest score = 6.85921%.
#
# resultat16.csv :
# nrows = 100M, 
# ...

