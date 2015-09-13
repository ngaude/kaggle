#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""
##################
# NOTE NOTE NOTE #
##################

ext = '.0' # default value

from utils import wdir,ddir,header,normalize_txt,add_txt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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
import time

def log_proba(df,vec,cla):
    assert 'txt' in df.columns
    X = vec.transform(df.txt)
    lp = cla.predict_log_proba(X)
    return (cla.classes_,lp)


(stage1_log_proba_test,stage3_log_proba_test) = joblib.load(ddir+'/joblib/log_proba_test'+ext)

##################
# bayes rulez ....
##################

def greedy_prediction(stage1_log_proba,stage3_log_proba):
    cat1 = [itocat1[c] for c in stage1_log_proba.argmax(axis=1)]
    for i in range(stage3_log_proba.shape[0]):
        stage3_log_proba[i,:] = [stage3_log_proba[i,j] if cat3tocat1[cat3]==cat1[i] else -666 for j,cat3 in enumerate(itocat3)]
    return

def bayes_prediction(stage1_log_proba,stage3_log_proba):
    for i in range(stage3_log_proba.shape[1]):
        cat3 = itocat3[i]
        cat1 = cat3tocat1[cat3]
        j = cat1toi[cat1]
        stage3_log_proba[:,i] += stage1_log_proba[:,j]

bayes_prediction(stage1_log_proba_test,stage3_log_proba_test)

proba_cat1_learner =  np.exp(np.max(stage1_log_proba_test,axis=1))
predict_cat1_test = [itocat1[i] for i in np.argmax(stage1_log_proba_test,axis=1)]
proba_cat3_learner =  np.exp(np.max(stage3_log_proba_test,axis=1))
predict_cat3_test = [itocat3[i] for i in np.argmax(stage3_log_proba_test,axis=1)]


import matplotlib.pyplot as plt

plt.hist(proba_cat1_learner,bins=300,label='cat1',alpha=0.5)
plt.hist(proba_cat3_learner,bins=300,label='cat3',alpha=0.5)
plt.legend()
plt.show()

resultat = pd.read_csv(ddir+'resultat60.csv',sep=';').fillna('')
resultat['score_cat1']= proba_cat1_learner
resultat['score_cat3']= proba_cat3_learner

test = pd.read_csv(ddir+'test.csv',sep=';').fillna('')
rayon = pd.read_csv(ddir+'rayon.csv',sep=';').fillna('ZZZ')
#rayon.Categorie3_Name = map(normalize_txt,rayon.Categorie3_Name.values)
#rayon.Categorie2_Name = map(normalize_txt,rayon.Categorie2_Name.values)
#rayon.Categorie1_Name = map(normalize_txt,rayon.Categorie1_Name.values)

df = test.merge(resultat,'left',None,'Identifiant_Produit','Id_Produit')
df = df.merge(rayon,'left',None,'Id_Categorie','Categorie3')
df = df.drop_duplicates()

weak = df[(df.score_cat1 < 0.2)]
weak = weak[['score_cat3','score_cat1','Categorie1_Name','Categorie2_Name','Categorie3_Name','Libelle','Marque','Description','prix']]

weak.to_csv(ddir+'weak.csv',sep=';',index=False)


pcat1 = [1.*sum(proba_cat1_learner > i)/len(proba_cat1_learner > i) for i in np.linspace(0,1,101)]
pcat3 = [1.*sum(proba_cat3_learner > i)/len(proba_cat3_learner > i) for i in np.linspace(0,1,101)]

plt.plot(pcat1,label='cat1',alpha=0.5)
plt.plot(pcat3,label='cat3',alpha=0.5)
plt.legend()
plt.show()


