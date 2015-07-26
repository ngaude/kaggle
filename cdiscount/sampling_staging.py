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

"""
import os
os.chdir('C:/Users/ngaude/Documents/GitHub/kaggle/cdiscount/')
"""

def score(df,vec,cla,target):
    X = vec.transform(iterText(df))
    Y = list(df[target])
    sc = cla.score(X,Y)
    return sc

def vectorizer(df):
    vec = TfidfVectorizer(
        min_df = 0.00009,
        stop_words = None,
        max_features=123456,
        smooth_idf=True,
        norm='l2',
        sublinear_tf=False,
        use_idf=True,
        ngram_range=(1,3))
    vec.fit(iterText(df))
    return vec

def classifier(df,vec,target):
    X = vec.transform(iterText(df))
    Y = list(df[target])
    cla = LogisticRegression() 
    cla.fit(X,Y)
    return cla

#######################
#######################
#######################
#######################
# create balanced data
# balanced : Categorie1
# balanced : Categorie3
#######################
#######################
#######################
#######################


def create_sample(dftrain,label,mincount,maxsampling):
    fname = ddir+'training_sampled_'+label+'.csv'
    dfsample = training_sample(dftrain,label,mincount,maxsampling)
    dfsample.to_csv(fname,sep=';',index=False,header=False)
    return

#dftrain = pd.read_csv(ddir+'training_shuffled_normed.csv',sep=';',names = header()).fillna('').reset_index()
#create_sample(dftrain,'Categorie1',25000,500)   #~1M rows
#create_sample(dftrain,'Categorie3',1000,50)     #~5M rows

#######################
#######################
#######################
#######################
# training
# staged : Categorie1 =>  Categorie3
#######################
#######################
#######################
#######################

dfvalid = pd.read_csv(ddir+'validation_normed.csv',sep=';',names = header()).fillna('').reset_index()

# stage 1 training : use Categorie1 sample training set
dftrain = pd.read_csv(ddir+'training_sampled_Categorie3.csv',sep=';',names = header()).fillna('').reset_index()

fname = ddir + 'joblib/stage1'
df = dftrain
vec = vectorizer(df)
cla = classifier(df,vec,"Categorie1")
labels = np.unique(df.Categorie1)
dfv = dfvalid
sct = score(df[:30000],vec,cla,'Categorie1')
scv = score(dfv,vec,cla,'Categorie1')
print '**********************************'
print 'classifier training score',sct
print 'classifier validation score',scv
print '**********************************'
joblib.dump((labels,vec,cla),fname)

del vec,cla

# stage 3 training : use Categorie3 sample training set

# Parallel training :
def training_stage3(ctx,cat):
    fname = ddir + 'joblib/stage3_'+str(cat)
    print '-'*50
    print 'training',basename(fname),':',cat
    df = dftrain[dftrain.Categorie1 == cat].reset_index()
    labels = np.unique(df.Categorie3)
    if len(labels)==1:
        print fname,'predict 100% ',labels[0]
        joblib.dump((labels,None,None),fname)
        scv = -1
        sct = -1
        return (sct,scv)
    print len(df)
    vec = vectorizer(df)
    cla = classifier(df,vec,"Categorie3")
    dfv = dfvalid[dfvalid.Categorie1 == cat].reset_index()
    sct = score(df[:30000],vec,cla,'Categorie3')
    if len(dfv)==0:
        scv = -1
    else:
        scv = score(dfv,vec,cla,'Categorie3')
    print '**********************************'
    print 'classifier',cat,'training score',sct
    print 'classifier',cat,'validation score',scv
    print '**********************************'
    joblib.dump((labels,vec,cla),fname)
    del vec,cla
    return (sct,scv)

dftrain = pd.read_csv(ddir+'training_sampled_Categorie3.csv',sep=';',names = header()).fillna('').reset_index()
cat1 = np.unique(dftrain.Categorie1)

mgr = Manager()
ctx = mgr.Namespace()
ctx.dftrain = dftrain
ctx.dfvalid = dfvalid

scs = Parallel(n_jobs=3)(delayed(training_stage3)(ctx,cat) for cat in cat1)

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

##################
# bayes rulez ....
##################

# P = P(cat3|cat1)*P(cat1)
################################
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


