#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils import ddir,header,add_txt,normalize_guess
import numpy as np
import pandas as pd
from os.path import basename,dirname
import time
import random
from sklearn.externals import joblib
from utils import itocat1,itocat2,itocat3
from utils import cat1toi,cat2toi,cat3toi
from utils import cat3tocat2,cat3tocat1,cat2tocat1
from utils import cat1count,cat2count,cat3count

#######################################################
# use complex Categorie3_Name as a good guess
# assuming Categorie1 is well-known
#######################################################

allowed_guess = [
'arceau securite poussette',
'beurre cacahuete',
'carte externe',
'centre repassage',
'combine cafetiere expresso',
'film protection',
'huile transmission',
'machine pop corn',
'pack accessoires',
'pack clavier souris',
'pad entrainement',
'papier dessin',
'porte casque',
'porte kayak',
'poubelle bord',
'recepteur infrarouge',
'serviette table',
'set sacs voyage',
'set valises',
'simulateur conduite',
'table mixage',
'telecommande domotique',
'tiroir bureau',
'umd',
]

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

rname  = '/home/ngaude/workspace/data/cdiscount.proba/resultat.auto.merging.csv'
rdir = dirname(rname)

resultat = pd.read_csv(rname,sep=';')

df = test.merge(resultat,'left',None,'Identifiant_Produit','Id_Produit')
df = df.merge(rayon,'left',None,'Id_Categorie','Categorie3')

df['guess'] = map(lambda (i,r):all_guess(r),df.iterrows())

diff = df[df.Categorie3 != df.guess]
diff = diff[['Identifiant_Produit','Description','Libelle','Marque','prix','Categorie3_Name','guess']]
diff = diff.merge(rayon,'left',None,'guess','Categorie3')
diff = diff[[u'guess',u'Categorie3_Name_x',u'Categorie3_Name_y',  u'Description', u'Libelle', u'Marque', u'prix']]
diff.to_csv(rdir+'/diff.csv',sep=';',index=False)

guess = df[['Id_Produit','guess']]
guess = guess.drop_duplicates()
guess.to_csv(rdir+'/guess.csv',sep=';',index=False)

