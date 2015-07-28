#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

from utils import wdir,ddir,header,normalize_file,iterText
from utils import MarisaTfidfVectorizer
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
import time

def vectorizer(txt):
    vec = TfidfVectorizer(
        min_df = 2,
        stop_words = None,
        max_features=123456,
        smooth_idf=True,
        norm='l2',
        sublinear_tf=True,
        use_idf=True,
        ngram_range=(1,2))
    X = vec.fit_transform(txt)
    return (vec,X)

def add_txt(df):
    assert 'Marque' in df.columns
    assert 'Libelle' in df.columns
    assert 'Description' in df.columns
    df['txt'] = (df.Marque+' ')*3+(df.Libelle+' ')*2+df.Description

def create_sample(df,label,mincount,maxsampling):
    fname = ddir+'training_sampled_'+label+'.csv'
    dfsample = training_sample(df,label,mincount,maxsampling)
    dfsample.to_csv(fname,sep=';',index=False,header=False)
    return dfsample

def training_stage1(dftrain,dfvalid):
    fname = ddir + 'joblib/stage1'
    df = dftrain
    dfv = dfvalid
    vec,X = vectorizer(df.txt)
    Y = df['Categorie1'].values
    dt = -time.time()
    cla = LogisticRegression(C=5)
    cla.fit(X,Y)
    dt += time.time()
    print 'training time',dt
    labels = np.unique(df.Categorie1)
    Xv = vec.transform(dfv.txt)
    Yv = dfv['Categorie1'].values
    sct = cla.score(X[:10000],Y[:10000])
    scv = cla.score(Xv,Yv)
    print '**********************************'
    print 'classifier training score',sct
    print 'classifier validation score',scv
    print '**********************************'
    joblib.dump((labels,vec,cla),fname)
    del X,Y,Xv,Yv,vec,cla
    return

def training_stage3(dftrain,dfvalid,cat):
    fname = ddir + 'joblib/stage3_'+str(cat)
    print '-'*50
    print 'training',basename(fname),':',cat
    df = dftrain[dftrain.Categorie1 == cat].reset_index(drop=True)
    dfv = dfvalid[dfvalid.Categorie1 == cat].reset_index(drop=True)
    labels = np.unique(df.Categorie3)
    if len(labels)==1:
        print fname,'predict 100% ',labels[0]
        joblib.dump((labels,None,None),fname)
        scv = -1
        sct = -1
        return (sct,scv)
    print 'samples=',len(df)
    vec,X = vectorizer(df.txt)
    Y = df['Categorie3'].values
    dt = -time.time()
    cla = LogisticRegression(C=5)
    cla.fit(X,Y)
    dt += time.time()
    print 'training time',cat,':',dt
    labels = np.unique(df.Categorie3)
    Xv = vec.transform(dfv.txt)
    Yv = dfv['Categorie3'].values
    sct = cla.score(X[:min(10000,len(df))],Y[:min(10000,len(df))])
    if len(dfv)==0:
        scv = -1
    else:
        scv = cla.score(Xv,Yv)
    print '**********************************'
    print 'classifier',cat,'training score',sct
    print 'classifier',cat,'validation score',scv
    print '**********************************'
    joblib.dump((labels,vec,cla),fname)
    del vec,cla
    return (sct,scv)

#####################
# create sample set 
# from training set
#####################

#dftrain = pd.read_csv(ddir+'training_shuffled_normed.csv',sep=';',names = header()).fillna('')
#create_sample(dftrain,'Categorie1',25000,500)   #~1M rows
#create_sample(dftrain,'Categorie3',200,10)     #~1M rows
#create_sample(dftrain,'Categorie3',1000,50)     #~5M rows
#del dftrain

#######################
# training
# stage1 : Categorie1 
# stage3 : Categorie3|Categorie1
#######################

dftrain = pd.read_csv(ddir+'training_sampled_Categorie3.csv',sep=';',names = header()).fillna('')
dfvalid = pd.read_csv(ddir+'validation_normed.csv',sep=';',names = header()).fillna('')
dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')

add_txt(dftrain)
add_txt(dfvalid)
add_txt(dftest)

dftrain = dftrain[['Categorie3','Categorie1','txt']]
dfvalid = dfvalid[['Categorie3','Categorie1','txt']]
dftest = dftest[['Identifiant_Produit','txt']]

# training stage1
training_stage1(dftrain,dfvalid)

# training parralel stage3
cat1 = np.unique(dftrain.Categorie1)
#training_stage3(ctx,cat1[0])
dfts = []
dfvs = []
for cat in cat1:
    dfts.append(dftrain[dftrain.Categorie1 == cat].reset_index(drop=True))
    dfvs.append(dfvalid[dfvalid.Categorie1 == cat].reset_index(drop=True))

scs = Parallel(n_jobs=2)(delayed(training_stage3)(dft,dfv,cat) for dft,dfv,cat in zip(dfts,dfvs,cat1))

#######################
# predicting
#######################

# predict : 
# P(Categorie3) = P(Categorie1) *  P(Categorie3|Categorie1)

def log_proba(df,vec,cla):
    X = vec.transform(iterText(df))
    lp = cla.predict_log_proba(X)
    return (cla.classes_,lp)

dfvalid = pd.read_csv(ddir+'validation_normed.csv',sep=';',names = header()).fillna('').reset_index()
dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')

df = dftest
#df = dfvalid

n = len(df)


#######################
# stage 1 log proba filling
#######################
stage1_log_proba = np.full(shape=(n,cat1count),fill_value = -666.,dtype = float)

fname = ddir + 'joblib/stage1'
(labels,vec,cla) = joblib.load(fname)
(classes,lp) = log_proba(df,vec,cla)
for i,k in enumerate(classes):
    j = cat1toi[k]
    stage1_log_proba[:,j] = lp[:,i]

del labels,vec,cla


#######################
# stage 3 log proba filling
#######################
stage3_log_proba = np.full(shape=(n,cat3count),fill_value = -666.,dtype = float)

for ii,cat in enumerate(itocat1):
    fname = ddir + 'joblib/stage3_'+str(cat)
    print '-'*50
    print 'predicting',basename(fname),':',ii,'/',len(itocat1)
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

if (df is dfvalid):
    fname = ddir+'/joblib/log_proba_valid'
else:
    fname = ddir+'/joblib/log_proba_test'

joblib.dump((stage1_log_proba,stage3_log_proba),fname)

##################
# (stage1_log_proba,stage3_log_proba) = joblib.load(fname)
##################

## FIXME
## FIXME
## FIXME
## >>>>>
#
## greedy approach:
#(stage1_log_proba,stage3_log_proba) = joblib.load(fname)
#
##predict_cat1 = [itocat1[i] for i in np.argmax(stage1_log_proba,axis=1)]
#
#for i,cat1 in enumerate(predict_cat1):
#    if i%1000==0:
#        print 1.*i/len(predict_cat1)
#    for j in [k for k,cat3 in enumerate(itocat3) if cat3tocat1[cat3] != cat1]:
#        stage3_log_proba[i,j] = -666
#
#predict_cat3 = [itocat3[i] for i in np.argmax(stage3_log_proba,axis=1)]
#
#score_cat1 = sum(dfvalid.Categorie1 == predict_cat1)*1.0/n
#score_cat3 = sum(dfvalid.Categorie3 == predict_cat3)*1.0/n
#print 'dfvalid scores =',score_cat1,score_cat3
#
## <<<<<
## FIXME
## FIXME
## FIXME

##################
# bayes rulez ....
##################

for i in range(stage3_log_proba.shape[1]):
    cat3 = itocat3[i]
    cat1 = cat3tocat1[cat3]
    j = cat1toi[cat1]
    stage3_log_proba[:,i] += stage1_log_proba[:,j]

predict_cat1 = [itocat1[i] for i in np.argmax(stage1_log_proba,axis=1)]
predict_cat3 = [itocat3[i] for i in np.argmax(stage3_log_proba,axis=1)]

def submit(df,Y):
    submit_file = ddir+'resultat.csv'
    df['Id_Produit']=df['Identifiant_Produit']
    df['Id_Categorie'] = Y
    df= df[['Id_Produit','Id_Categorie']]
    df.to_csv(submit_file,sep=';',index=False)

if df is dfvalid:
    score_cat1 = sum(dfvalid.Categorie1 == predict_cat1)*1.0/n
    score_cat3 = sum(dfvalid.Categorie3 == predict_cat3)*1.0/n
    print 'dfvalid scores =',score_cat1,score_cat3
else:
    submit(df,predict_cat3)

# resultat22.csv scored 58,43218% 
# resultat23.csv scored 52.95399% (2 sampling : for Categorie1 (49*25000) & for Categorie3 (4546*1000) )
# resultat24.csv scored 61,54948% ( 1000 samples (4546 classes ) with staged 1&3 proba logit)
# resultat29.csv scored 65,78993% (1000samples/4546classes staged 1&3 optimized tfidf-vectorizer and L2-regularization)

