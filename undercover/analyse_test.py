#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils import ddir,header,add_txt,normalize_guess
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
dfs.append(pd.read_csv(ddir+'resultat44.csv',sep=';'))
dfs.append(pd.read_csv(ddir+'resultat45.csv',sep=';'))
dfs.append(pd.read_csv(ddir+'resultat46.csv',sep=';'))
dfs.append(pd.read_csv(ddir+'resultat47.csv',sep=';'))
dfs.append(pd.read_csv(ddir+'resultat48.csv',sep=';'))
dfs.append(pd.read_csv(ddir+'resultat49.csv',sep=';'))

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
# compare with categorie distribution
############################################

#df = pd.read_csv(ddir+'training_sampled_Categorie3_200.csv',sep=';',names = header()).fillna('')
#df = pd.read_csv(ddir+'validation_perfect.csv',sep=';',names = header()).fillna('')
df = pd.read_csv(ddir+'resultat44.csv',sep=';').fillna('')

if 'Categorie3' in df.columns:
    label = 'Categorie3'
    target = 'Identifiant_Produit'

if 'Id_Categorie' in df.columns:
    label = 'Id_Categorie'
    target = 'Id_Produit'



ccv = df.groupby(label)[target].count()
ccv.sort(ascending=False)
ccv = [ccv.get(cat,0) for cat in ccr.index]
eps = 0.000001
ccv = (ccv-np.mean(ccv)+eps)/(np.std(ccv)+eps)

plt.plot(ccr,label = 'resultat')
plt.plot(ccv,label = 'validation')
plt.legend()
plt.show()


############################################
# compare with categorie distribution
############################################

allowed_guess = [
'canne peche',
'porte casque',
'poubelle bord',
'porte kayak',
'bac litiere',
'pack accessoires',
'set sacs voyage',
'set valises',
'recepteur infrarouge',
'simulateur conduite',
'beurre cacahuete',
'machine pop corn',
'lits superposes',
'pate tartiner',
'pate modeler',
'centre repassage',
'machine expresso',
'boisson recuperation',
'combine cafetiere expresso',
'station accueil',
'table mixage',
'pad entrainement',
'boite rangement',
'enfile aiguille',
'huile transmission',
'machine fumee',
'meuble range',
'tiroir bureau',
'arceau securite poussette',
'pack logiciel',
'carte externe',
'disque dur externe',
'papier dessin',
'umd',
'dvd',
'blu-ray',
'serviette table',
'cadre photo',
'coussin chauffant',
'physique chimie',
'histoire geographie',
'pack clavier souris',
'telecommande domotique',
'enfile aiguille',
'nettoyant vitres',
'film protection']

# 
# def all_guess(r):
#     rdf = rayon[rayon.Categorie1 == r.Categorie1]
#     filt = [one_guess(r.txt,name) for name in rdf.Categorie3_Name.values]
#     guess = rdf.Categorie3[filt].values
#     guess_name = rdf.Categorie3_Name[filt].values
#     if len(guess)==1 and len(guess_name[0].split())>1 and guess_name[0] not in forbidden_guess:
#         return guess[0]
#     return r.Categorie3
# 
# def one_guess(txt,name):
#     name = name.split()
#     if len(name)==0:
#         return False
#     for w in name:
#         if ' '+w+' ' not in txt:
#             return False
#     return True

def all_guess(r):
    rdf = rayon[rayon.Categorie1 == r.Categorie1]
    filt = [one_guess(r.txt,name) for name in rdf.Categorie3_Name.values]
    guess = rdf.Categorie3[filt].values
    if len(guess)==1: 
        return guess[0]
    return r.Categorie3

def one_guess(txt,name):
    if name not in allowed_guess:
        return False
    if name in txt:
        return True
    return False

# join rayon.csv & test.csv & resultat44.csv
# keep id

rayon = pd.read_csv(ddir+'rayon.csv',sep=';').fillna('ZZZ')
rayon.Categorie3_Name = map(normalize_guess,rayon.Categorie3_Name.values)
rayon.Categorie2_Name = map(normalize_guess,rayon.Categorie2_Name.values)
rayon.Categorie1_Name = map(normalize_guess,rayon.Categorie1_Name.values)

test = pd.read_csv(ddir+'test.csv',sep=';').fillna('')
add_txt(test)
test.txt = map(normalize_guess,test.txt)

resultat = pd.read_csv(ddir+'resultat44.csv',sep=';')


df = test.merge(resultat,'left',None,'Identifiant_Produit','Id_Produit')
df = df.merge(rayon,'left',None,'Id_Categorie','Categorie3')

df['guess'] = map(lambda (i,r):all_guess(r),df.iterrows())

diff = df[df.Categorie3 != df.guess]
diff = diff[['Identifiant_Produit','Description','Libelle','Marque','prix','Categorie3_Name','guess']]
diff = diff.merge(rayon,'left',None,'guess','Categorie3')
diff = diff[[u'guess',u'Categorie3_Name_x',u'Categorie3_Name_y',  u'Description', u'Libelle', u'Marque', u'prix']]
diff.to_csv(ddir+'diff.csv',sep=';',index=False)

guess = df[['Id_Produit','guess']]
guess = guess.drop_duplicates()
guess.to_csv(ddir+'guess.csv',sep=';',index=False)

