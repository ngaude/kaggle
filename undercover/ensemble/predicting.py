#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

from utils import wdir,ddir,header,normalize_file,iterText
from utils import MarisaTfidfVectorizer
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from os.path import basename

from utils import itocat1,itocat2,itocat3
from utils import cat1toi,cat2toi,cat3toi
from utils import cat3tocat2,cat3tocat1,cat2tocat1
from utils import cat1count,cat2count,cat3count
from utils import training_sample
from os.path import isfile

from joblib import Parallel, delayed
from multiprocessing import Manager

def submit(df,Y):
    submit_file = ddir+'resultat.csv'
    df['Id_Produit']=df['Identifiant_Produit']
    df['Id_Categorie'] = Y
    df= df[['Id_Produit','Id_Categorie']]
    df.to_csv(submit_file,sep=';',index=False)

#######################
#######################
#######################
#######################
# predicting
#######################
#######################
#######################
#######################

# staged : Categorie1 =>  Categorie3
# using sum log prob 1&3 to estimate total proba

def log_proba(df,vec,cla):
    X = vec.transform(iterText(df))
    lp = cla.predict_log_proba(X)
    return (cla.classes_,lp)

#######################
# stage 1 log proba filling
#######################

def log_proba_stage1(df,ensemble):
    stage1_log_proba = np.full(shape=(len(df),cat1count),fill_value = -666.,dtype = float)
    fname = ddir + 'joblib/stage1.'+str(ensemble)
    (labels,vec,cla) = joblib.load(fname)
    (classes,lp) = log_proba(df,vec,cla)
    for i,k in enumerate(classes):
        j = cat1toi[k]
        stage1_log_proba[:,j] = lp[:,i]
    del labels,vec,cla
    return stage1_log_proba


#######################
# stage 3 log proba filling
#######################

def log_proba_stage3(df,ensemble):
    stage3_log_proba = np.full(shape=(len(df),cat3count),fill_value = -666.,dtype = float)
    for ii,cat in enumerate(itocat1):
        fname = ddir + 'joblib/stage3_'+str(cat)+'.'+str(ensemble)
        print '-'*50
        print 'log_probaing',basename(fname),':',ii,'/',len(itocat1)
        if not isfile(fname): 
            continue
        (labels,vec,cla) = joblib.load(fname)
        if len(labels)==1:
            k = labels[0]
            j = cat3toi[k]
            stage3_log_proba[:,j] = 0
            continue
        (classes,lp) = log_proba(df,vec,cla)
        for i,k in enumerate(classes):
            j = cat3toi[k]
            stage3_log_proba[:,j] = lp[:,i]
        del labels,vec,cla
    return stage3_log_proba

def log_proba_stage(df,ensemble,test=False):
    a = -time.time()
    stage1_log_proba = log_proba_stage1(df,ensemble)
    stage3_log_proba = log_proba_stage3(df,ensemble)
    if (test):
        fname = ddir+'/joblib/log_proba_test.'+str(ensemble)
    else:
        fname = ddir+'/joblib/log_proba_valid.'+str(ensemble)
    joblib.dump((stage1_log_proba,stage3_log_proba),fname)
    a += time.time()
    print 'ensemble',ensemble,'time =',a
    del stage1_log_proba,stage3_log_proba
    return

dfvalid = pd.read_csv(ddir+'validation_normed.csv',sep=';',names = header()).fillna('').reset_index()
dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')

#Parallel(n_jobs=4)(delayed(log_proba_stage)(dfvalid,ensemble,test=False) for ensemble in range(0,10))
#Parallel(n_jobs=4)(delayed(log_proba_stage)(dftest,ensemble,test=True) for ensemble in range(0,10))

# ensemble 0 : dfvalid scores = 0.829303640009 0.669404120673
# ensemble 1 : dfvalid scores = 0.827323011521 0.670442742929
# ensemble 2 : dfvalid scores = 0.827757783628 0.67150551919
# ensemble 3 : dfvalid scores = 0.828844713896 0.670877515036
# ensemble 4 : dfvalid scores = 0.829158715973 0.672350909398
# ensemble 5 : dfvalid scores = 0.827951015676 0.670587666965
# ensemble 6 : dfvalid scores = 0.829641796092 0.670249510881
# ensemble 7 : dfvalid scores = 0.828192555735 0.672036907321
# ensemble 8 : dfvalid scores = 0.828965483926 0.671674597232
# ensemble 9 : dfvalid scores = 0.827902707664 0.670829207024
# ---------- : ----------------------------------------------
# overall    : dfvalid scores = 0.829931644163 0.674283229874
# majority   : dfvalid scores = 0.829738412116 0.674355691892

##################
# (stage1_log_proba,stage3_log_proba) = joblib.load(fname)
##################

def predicting(s1,s3):
    ##################
    # bayes rulez ....
    ##################
    # P = P(cat3|cat1)*P(cat1)
    ##########################
    s = s3.copy() 
    for i in range(s3.shape[1]):
        cat3 = itocat3[i]
        cat1 = cat3tocat1[cat3]
        j = cat1toi[cat1]
        s[:,i] += s1[:,j]
    p1 = [itocat1[i] for i in np.argmax(s1,axis=1)]
    p3 = [itocat3[i] for i in np.argmax(s,axis=1)]
    return p1,p3

def overall_predicting(df):
    stage1_log_proba = np.zeros(shape=(len(df), cat1count))
    stage3_log_proba = np.zeros(shape=(len(df), cat3count))
    for ensemble in range(10):
        if (df is dfvalid):
            fname = ddir+'/joblib/log_proba_valid.'+str(ensemble)
        else:
            fname = ddir+'/joblib/log_proba_test.'+str(ensemble)
        print 'reading',fname
        s1,s3 = joblib.load(fname)
        stage1_log_proba += s1
        stage3_log_proba += s3
        del s1,s3
    (p1,p3) = predicting(stage1_log_proba,stage3_log_proba)
    return (p1,p3)

from collections import Counter

def majority_predicting(df):
    p1s = np.ndarray(shape=(len(df),10),dtype=object)
    p3s = np.ndarray(shape=(len(df),10),dtype=object)
    for ensemble in range(10):
        if (df is dfvalid):
            fname = ddir+'/joblib/log_proba_valid.'+str(ensemble)
        else:
            fname = ddir+'/joblib/log_proba_test.'+str(ensemble)
        print 'reading',fname
        s1,s3 = joblib.load(fname)
        (p1,p3) = predicting(s1,s3)
        p1s[:,ensemble] = p1
        p3s[:,ensemble] = p3
        del s1,s3,p1,p3
    p1 = [None]*len(df)
    p3 = [None]*len(df)
    for i in range(len(df)):
        freq = {v:k for k,v in Counter(p1s[i,:]).iteritems()}
        p1[i]=freq[max(freq)]
        freq = {v:k for k,v in Counter(p3s[i,:]).iteritems()}
        p3[i]=freq[max(freq)]
    return (p1,p3)

#df = dfvalid
df = dftest
(predict_cat1,predict_cat3) = majority_predicting(df)


if df is dfvalid:
    score_cat1 = sum(dfvalid.Categorie1 == predict_cat1)*1.0/len(df)
    score_cat3 = sum(dfvalid.Categorie3 == predict_cat3)*1.0/len(df)
    print 'dfvalid scores =',score_cat1,score_cat3
else:
    submit(df,predict_cat3)


# resultat22.csv scored 58,43218% 
# resultat23.csv scored 52.95399% (2 sampling : for Categorie1 (49*25000) & for Categorie3 (4546*1000) )
# resultat24.csv scored 61,54948% ( 1000 samples (4546 classes ) with staged 1&3 proba logit)
# resultat27.csv scored 61,81308% (overall predict from 10x 1000 samples from 4546 classes with stage 1&3 logit)
# resultat28.csv scored 61,81308% (majority predict from 10x 1000 samples from 4546 classes with stage 1&3 logit)


