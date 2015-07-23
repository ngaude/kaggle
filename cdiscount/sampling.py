# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 22:12:51 2015

@author: ngaude
"""

import os
os.chdir('C:/Users/ngaude/Documents/GitHub/kaggle/cdiscount/')


from utils import wdir,ddir,header,normalize_file
from utils import MarisaTfidfVectorizer,iterText
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from os.path import basename

def get_sample(mincount = 1000):
    dftrain = pd.read_csv(ddir+'training_shuffled_normed.csv',sep=';',names = header()).fillna('')
    c3 = dftrain.Categorie3
    cc = c3.groupby(c3)
    s = (cc.count() > mincount)
    cat3maj = s[s].index
    dfs = []
    for i,cat in enumerate(cat3maj):
        if i%10==0:
            print i,'/',len(cat3maj),':'
        df = dftrain[dftrain.Categorie3 == cat]
        rows = random.sample(df.index, mincount)
        dfs.append(df.ix[rows])
    dfsample = pd.concat(dfs)
    dfsample.to_csv(ddir+'training_sampled.csv',sep=';',header=False,index = False)
    return

def score(df,vec,cla,target):
    X = vec.transform(iterText(df))
    Y = list(df[target])
    sc = cla.score(X,Y)
    return sc

def vectorizer(df):
    # 1M max_features should fit in memory, 
    # OvA will be at max 184 classes, 
    # so we can fit coef_ =  1M*184*8B ~ 1GB in memory easily
    vec = MarisaTfidfVectorizer(
        min_df = 1,
        stop_words = None,
        max_features=1000000,
        smooth_idf=True,
        norm='l2',
        sublinear_tf=True,
        use_idf=True,
        ngram_range=(1,3))
    vec.fit(iterText(df))
    return vec

def classifier(df,vec,target):
    X = vec.transform(iterText(df))
    Y = list(df[target])
    cla = LogisticRegression() 
    cla.fit(X,Y)
    print 'classifier training score',cla.score(X,Y)
    return cla

def predict(df,vec,cla):
    X = vec.transform(iterText(df))
    Y = cla.predict(X)
    return Y

def submit(df,Y):
    submit_file = ddir+'resultat.csv'
    df['Id_Produit']=df['Identifiant_Produit']
    df['Id_Categorie'] = Y
    df= df[['Id_Produit','Id_Categorie']]
    df.to_csv(submit_file,sep=';',index=False)


##################################
# create sample set

# get_sample() 

##################################
# train model on sample set
    
dfsample = pd.read_csv(ddir+'training_sampled.csv',sep=';',names = header()).fillna('')
dfvalid = pd.read_csv(ddir+'validation_normed.csv',sep=';',names = header()).fillna('').reset_index()
dfvalid = pd.read_csv(ddir+'validation_normed.csv',sep=';',names = header()).fillna('')

fname = ddir + 'joblib/logit'
df = dfsample
vec = vectorizer(df)
cla = classifier(df,vec,"Categorie3")
labels = np.unique(df.Categorie3)
dfv = dfvalid
sc = score(dfv,vec,cla,"Categorie3")
print '**********************************'
print 'classifier',basename(fname),'valid score',sc
print '**********************************'
joblib.dump((labels,vec,cla),fname)
del vec,cla

##################################
# predict on test set

fname = ddir + 'joblib/logit'
(labels,vec,cla) = joblib.load(fname)
df = dftest
Y = predict(df,vec,cla)
submit(df,Y)



