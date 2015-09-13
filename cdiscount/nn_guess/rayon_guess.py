#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils import ddir,header,add_txt
import numpy as np
import pandas as pd
from os.path import basename,dirname,isfile
import time
import random
from sklearn.externals import joblib
from utils import itocat1,itocat2,itocat3
from utils import cat1toi,cat2toi,cat3toi
from utils import cat3tocat2,cat3tocat1,cat2tocat1
from utils import cat1count,cat2count,cat3count
import sys

assert len(sys.argv) == 2  ##### usage guess.py $PROBA.CSV ####
assert isfile(sys.argv[1]) ##### usage guess.py $PROBA.CSV ####

pname  = sys.argv[1]
# pname = 'proba.auto.merging.15.csv'
pdir = dirname(pname)
##################
# FIXME : ensure that confidence level are the same between logistic regression proba et guessing proba
# proba_score = 0.6768667
# <==>
# sum(proba.Proba_Categorie3)/len(df) # 0.7525964785959941
##################

rayon = pd.read_csv(ddir+'rayon.csv',sep=';')
test = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(True)).fillna('')
add_txt(test)
proba = pd.read_csv(pname,sep=';')
df = test.merge(proba,'left',None,'Identifiant_Produit','Id_Produit')
df = df.merge(rayon,'left',None,'Id_Categorie','Categorie3')

rg = pd.read_csv(ddir+'rayon_guessing.csv',sep=';')
g = rg.groupby('Categorie1')

guess_correction = 0
num_correction = 0

best_Categorie3 = df.Categorie3.values
#best_Categorie3 = [1000015309]*len(df)

for i,r in df.iterrows():
    rdf = g.get_group(r.Categorie1)
    candidates = rdf[[t in r.txt for t in rdf.guess]].reset_index(drop=True)
    if len(candidates)==0:
        continue
    most_probable = candidates.iloc[candidates.Proba_guess.argmax()]
    if (most_probable.Proba_guess > df.iloc[i].Proba_Categorie3) and (most_probable.Categorie3 != df.iloc[i].Categorie3) and (most_probable.Proba_guess):
        guess_correction += most_probable.Proba_guess - df.iloc[i].Proba_Categorie3
        num_correction += 1
        print i,'/',len(df),'(#',num_correction,')guess correction of', guess_correction
        best_Categorie3[i] = most_probable.Categorie3

df['Id_Categorie'] = best_Categorie3
guess = df[['Id_Produit','Id_Categorie']]
guess = guess.drop_duplicates()
guess.to_csv(pdir+'resultat.rayon_guess.csv',sep=';',index=False)
