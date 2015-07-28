#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.externals import joblib
import pandas as pd

#ddir = 'E:/workspace/data/cdiscount/'
#wdir = 'C:/Users/ngaude/Documents/GitHub/kaggle/cdiscount/'
ddir = '/home/ngaude/workspace/data/cdiscount/'
wdir = '/home/ngaude/workspace/github/kaggle/cdiscount/'

Xtest = joblib.load(ddir+'joblib/Xtest')
classifier = joblib.load(ddir+'joblib/classifier')
(itocat1,cat1toi,itocat2,cat2toi,itocat3,cat3toi) = joblib.load(ddir+'joblib/itocat')

print 'predicting >>>'
Ytest = classifier.predict(Xtest)
print 'predicting <<<'
cat3 = map(lambda i:itocat3[i],Ytest)


submit_file = ddir+'resultat11.csv'
print 'writing >>>',submit_file
fname = ddir + 'test_shuffled.csv'
names = ['Identifiant_Produit','Description','Libelle','Marque','prix']
df_test = pd.read_csv(fname,sep=';',names=names)

df_test['Id_Produit']=df_test['Identifiant_Produit']
df_test['Id_Categorie'] = cat3
df_test = df_test[['Id_Produit','Id_Categorie']]

df_test.to_csv(submit_file,sep=';',index=False)
print 'writing <<<'


(Xneighbor,Yneighbor) = joblib.load(ddir+'joblib/XYneighbor')
print 'train score = ',classifier.score(Xneighbor,Yneighbor[:,2])


