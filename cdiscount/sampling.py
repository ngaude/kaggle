# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 22:12:51 2015

@author: ngaude
"""

"""
import os
os.chdir('C:/Users/ngaude/Documents/GitHub/kaggle/cdiscount/')
"""

from utils import wdir,ddir,header,normalize_file,iterText
from utils import MarisaTfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from os.path import basename

def get_sample(mincount = 200):
    dftrain = pd.read_csv(ddir+'training_shuffled_normed.csv',sep=';',names = header()).fillna('')
    c3 = dftrain.Categorie3
    cc = c3.groupby(c3)
    s = (cc.count() > mincount/10)
    cat3maj = s[s].index
    dfs = []
    for i,cat in enumerate(cat3maj):
        if i%10==0:
            print i,'/',len(cat3maj),':'
        df = dftrain[dftrain.Categorie3 == cat]
        if len(df)>=mincount:
            # undersample mincount samples
            rows = random.sample(df.index, mincount)
            dfs.append(df.ix[rows])
        else:
            # sample all samples + oversample the remaining
            dfs.append(df)
            df = df.iloc[np.random.randint(0, len(df), size=mincount-len(df))]
            dfs.append(df)
    dfsample = pd.concat(dfs)
    dfsample = dfsample.reset_index()
    dfsample.reindex(np.random.permutation(dfsample.index))
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

if not os.path.isfile(ddir+'training_sampled.csv'):
    get_sample() 

##################################
# train model on sample set

dfsample = pd.read_csv(ddir+'training_sampled.csv',sep=';',names = header()).fillna('')
dfvalid = pd.read_csv(ddir+'validation_normed.csv',sep=';',names = header()).fillna('')
dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')


df = dfsample

fname = ddir + 'joblib/vectorizer'
if not os.path.isfile(fname):
    vec = vectorizer(df)
    joblib.dump(vec,fname)
else:
    vec = joblib.load(ddir+fname)

fname = ddir + 'joblib/classifier'
if not os.path.isfile(fname):
    cla = classifier(df,vec,'Categorie3')
    labels = np.unique(df.Categorie3)
    joblib.dump((labels,cla),fname)
else:
    (labels,cla) = joblib.load(fname)

sct = score(dfsample[:30000],vec,cla,'Categorie3')
scv = score(dfvalid,vec,cla,'Categorie3')

print '**********************************'
print 'classifier training score',sct
print 'classifier validation score',scv
print '**********************************'

# del vec,cla

##################################
# predict on test set

df = dftest
Y = predict(df,vec,cla)
submit(df,Y)

# resultat21.csv
# classifier training score 0.8347
# classifier validation score 0.564889736963
# classifier test score 0,5487365



