#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils import header,add_txt
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

ddir = '/home/ngaude/workspace/data/cdiscount.proba/'

assert len(sys.argv) == 2  ##### usage guess.py $RESULTAT.CSV ####
rname  = sys.argv[1]
assert isfile(ddir+rname) ##### usage guess.py $RESULTAT.CSV ####


test_normed = pd.read_csv(ddir+'test_normed.csv',sep=';',names=header(True)).fillna('')
add_txt(test_normed)
test_num_word = map(lambda t:len(set(t.split())),test_normed.txt)

test_nn = pd.read_csv(ddir+'test_nn.csv',sep=';').fillna('')
test_nn['Marque'] = test_nn.Marque_nn
test_nn['Libelle'] = test_nn.Libelle_nn
test_nn['Description'] = test_nn.Description_nn
add_txt(test_nn)
nn_num_word = map(lambda t:len(set(t.split())),test_nn.txt)
test_nn.drop('Marque', axis=1, inplace=True)
test_nn.drop('Libelle', axis=1, inplace=True)
test_nn.drop('Description', axis=1, inplace=True)

best = pd.read_csv(ddir+rname,sep=';')
#best = pd.read_csv('proba.auto.merging.60.csv',sep=';')
#best.Id_Categorie = 1000015309

nn = test_nn.merge(best,'left',None,'Identifiant_Produit_test','Id_Produit')
nn['test_num_word'] = test_num_word;
nn['nn_num_word'] = nn_num_word;

def nn_guess_func_ultra_conservative(r):
#    if r.Produit_Cdiscount==0:
#         return r.Id_Categorie
    if (r.D < 0.05) and (r.test_num_word)>=4 and (r.nn_num_word) >=4:
        return r.Categorie3
    if (r.D < 0.07) and (r.test_num_word)>=6 and (r.nn_num_word) >=6:
        return r.Categorie3
    return r.Id_Categorie


def nn_guess_func_conservative(r):
    if r.Produit_Cdiscount==0:
         return r.Id_Categorie
    if (r.D < 0.05) and (r.test_num_word)>=4 and (r.nn_num_word) >=4:
        return r.Categorie3
    if (r.D < 0.07) and (r.test_num_word)>=6 and (r.nn_num_word) >=6:
        return r.Categorie3
    if (r.D < 0.10) and (r.test_num_word)>=7 and (r.nn_num_word) >=7:
        return r.Categorie3
    if (r.D < 0.12) and (r.test_num_word)>=12 and (r.nn_num_word) >=11:
        return r.Categorie3
    if (r.D < 0.14) and (r.test_num_word)>=16 and (r.nn_num_word) >=16:
        return r.Categorie3
    if (r.D < 0.18) and (r.test_num_word)>=17 and (r.nn_num_word) >=17:
        return r.Categorie3
    return r.Id_Categorie

# def nn_guess_func(r):
#     if (r.D < 0.1) and (r.test_num_word) >=3:
#         return r.Categorie3
#     if (r.D < 0.13) and (r.test_num_word) >= 4:
#         return r.Categorie3
#     if (r.D < 0.15) and (r.test_num_word) >= 5:
#         return r.Categorie3
#     if (r.D < 0.18) and (r.test_num_word) >= 6:
#         return r.Categorie3
#     if (r.D < 0.20) and (r.test_num_word) >= 7:
#         return r.Categorie3
#     if (r.D < 0.21) and (r.test_num_word) >= 8:
#         return r.Categorie3
#     return r.Id_Categorie

nn['guess'] = nn.apply(nn_guess_func_ultra_conservative,axis=1)

print 'nn correction = ', sum(nn.Id_Categorie != nn.guess)
print 'D median = ',nn[(nn.Id_Categorie != nn.guess)].D.median()

### save diff for analysis ...
diff = nn[nn.guess != nn.Id_Categorie]
diff = diff[['Produit_Cdiscount','Identifiant_Produit_test','Marque_test','Libelle_test','Description_test','Categorie3','D','test_num_word','Id_Categorie','Marque_nn','Libelle_nn','Description_nn','Identifiant_Produit_nn']]
rayon = pd.read_csv(ddir+'rayon.csv',sep=';')
diff = diff.merge(rayon,'left','Categorie3')
diff = diff.merge(rayon,'left',None,'Id_Categorie','Categorie3',suffixes=('_nn','_lr'))

diff = diff[['Produit_Cdiscount','Identifiant_Produit_test','Marque_test', 'Libelle_test', 'Description_test','Categorie3_Name_lr','Categorie3_Name_nn','D','test_num_word','Marque_nn','Libelle_nn','Description_nn','Identifiant_Produit_nn']]
diff.to_csv('nn_diff.csv',sep=';',index=False)

###################################
# merge guess with results        #
###################################

nn.Id_Categorie = nn.guess
nn = nn[['Id_Produit','Id_Categorie']]
nn = nn.sort('Id_Produit')
nn.to_csv(ddir+'resultat.nn_guess.csv',sep=';',index=False)

