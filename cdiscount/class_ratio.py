#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils import ddir
import numpy as np
import pandas as pd
from sklearn.externals import joblib


############################################
# top test submission categorie distribution
############################################

dfs = []
dfs.append(pd.read_csv(ddir+'resultat53.csv',sep=';'))
dfs.append(pd.read_csv(ddir+'resultat51.csv',sep=';'))
dfs.append(pd.read_csv(ddir+'resultat49.csv',sep=';'))
dfs.append(pd.read_csv(ddir+'resultat47.csv',sep=';'))
dfs.append(pd.read_csv(ddir+'resultat45.csv',sep=';'))
dfs.append(pd.read_csv(ddir+'resultat44.csv',sep=';'))
dfs.append(pd.read_csv(ddir+'resultat39.csv',sep=';'))
dfs.append(pd.read_csv(ddir+'resultat36.csv',sep=';'))

df = pd.concat(dfs)

ccr = df.groupby('Id_Categorie').Id_Produit.count()
ccr.sort(ascending=False)
ccr = (ccr-np.mean(ccr))/np.std(ccr)

################################################
# compute a reasonnable sampling value for class 
################################################

# normalize for 99% of classes
pmin = np.percentile(ccr,1)
pmax = np.percentile(ccr,99)

ccr = np.clip(ccr,pmin,pmax)

class_ratio = np.power(2,ccr)
class_ratio = (1+ccr/2)

joblib.dump(class_ratio,ddir+'joblib/class_ratio')
