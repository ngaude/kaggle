# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

from utils import wdir,ddir,header,normalize_file
from utils import iterText
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from os.path import basename

def cat_freq(df):
    # compute Categorie3 classe frequency
    g = df.groupby('Categorie3').Libelle.count()
    mfreq = max(g)*1.
    g = 1./g*mfreq
    return dict(g)

nrows = 1000000
dftrain = pd.read_csv(ddir+'training_shuffled_normed.csv',sep=';',names = header(),nrows=nrows).fillna('')
dfvalid = pd.read_csv(ddir+'validation_normed.csv',sep=';',names = header()).fillna('').reset_index()
dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names=header(test=True)).fillna('')

vec = TfidfVectorizer(
    min_df = 0.00005,
    max_features=123456,
    stop_words=None,
    strip_accents = 'unicode',
    smooth_idf=True,
    norm='l2',
    sublinear_tf=True,
    use_idf=True,
    ngram_range=(1,3))

vec = vec.fit(iterText(dftest))

X = vec.transform(iterText(dftrain))
Y = dftrain.Categorie3
W = cat_freq(dftrain)

joblib.dump((X,Y,W),ddir+'joblib/XYWtrain')
joblib.dump(vec,ddir+'joblib/vectorizer')

##########################################################
# train a SGD classifier 
# on X,Y training samples 
# balanced by W
##########################################################

# (X,Y,W) = joblib.load(ddir+'joblib/XYWtrain')
# vec = joblib.load(ddir+'joblib/vectorizer')

cl = SGDClassifier(class_weight = 'auto',n_jobs=4)
cl.fit(X,Y)
cl.sparsify()

joblib.dump(cl,ddir+'joblib/classifier')

##########################################################
# simple prediction
##########################################################

cl = joblib.load(ddir+'joblib/classifier')

Xvalid = vec.transform(iterText(dfvalid))
Yvalid = dfvalid.Categorie3
Xtest = vec.transform(iterText(dftest))

print 'training score', cl.score(X[:50000],Y[:50000])
print 'validation score',cl.score(Xvalid,Yvalid)

predict_cat3 = cl.predict(Xtest)

def submit(df,prediction):
    submit_file = ddir+'resultat.csv'
    df['Id_Produit']=df['Identifiant_Produit']
    df['Id_Categorie'] = prediction
    df= df[['Id_Produit','Id_Categorie']]
    df.to_csv(submit_file,sep=';',index=False)

submit(dftest,predict_cat3)

