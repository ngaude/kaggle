#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils import ddir,header
import numpy as np
import pandas as pd
from os.path import basename
import time
import random
from sklearn.externals import joblib
from utils import itocat1,itocat2,itocat3
from utils import cat1toi,cat2toi,cat3toi
from utils import cat3tocat2,cat3tocat1,cat2tocat1
from utils import cat1count,cat2count,cat3count

import matplotlib.pyplot as plt

############################################
# top test submission categorie distribution
############################################

dfs = []
dfs.append(pd.read_csv(ddir+'resultat29.csv',sep=';'))
dfs.append(pd.read_csv(ddir+'resultat31.csv',sep=';'))
dfs.append(pd.read_csv(ddir+'resultat33.csv',sep=';'))
dfs.append(pd.read_csv(ddir+'resultat35.csv',sep=';'))
dfs.append(pd.read_csv(ddir+'resultat36.csv',sep=';'))
dfs.append(pd.read_csv(ddir+'resultat37.csv',sep=';'))
dfs.append(pd.read_csv(ddir+'resultat38.csv',sep=';'))

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

plt.plot(class_ratio)
plt.show()



############################################
# validation categorie distribution
############################################

#dfvalid = pd.read_csv(ddir+'training_sampled_Categorie3_200.csv',sep=';',names = header()).fillna('')
dfvalid = pd.read_csv(ddir+'validation_perfect.csv',sep=';',names = header()).fillna('')


ccv = dfvalid.groupby('Categorie3').Identifiant_Produit.count()
ccv.sort(ascending=False)
ccv = [ccv.get(cat,0) for cat in ccr.index]
eps = 0.000001
ccv = (ccv-np.mean(ccv)+eps)/(np.std(ccv)+eps)

plt.plot(ccr,label = 'resultat')
plt.plot(ccv,label = 'validation')
plt.legend()
plt.show()

