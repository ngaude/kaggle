#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils import header,add_txt,normalize_guess
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

ddir = '/home/ngaude/workspace/data/cdiscount/'

assert len(sys.argv) == 2  ##### usage guess.py $RESULTAT.CSV ####

rname  = ddir+sys.argv[1]

assert isfile(rname) ##### usage guess.py $RESULTAT.CSV ####



#######################################################
# use complex Categorie3_Name as a good guess
# assuming Categorie1 is well-known
#######################################################

with open(ddir+'allowed_guess_optimized.txt') as f:
    allowed_guess = [guess for guess in f.read().split('\n') if len(guess)>5]


def all_guess(i,r):
    if (i%1000)==0:
        print i
    if r.Categorie1 not in gg.groups:
        return r.Id_Categorie
    rdf = gg.get_group(r.Categorie1)
    filt = [one_guess(r.txt,name) for name in rdf.Categorie3_Name.values]
    guess = rdf.Categorie3[filt].values
    if len(guess)==1: 
        return guess[0]
    return r.Id_Categorie

def one_guess(txt,name):
    if name not in allowed_guess:
        return False
    if name in txt:
        return True
    return False


rayon = pd.read_csv(ddir+'rayon.csv',sep=';').fillna('ZZZ')
rayon.Categorie3_Name = map(normalize_guess,rayon.Categorie3_Name.values)
rayon = rayon[rayon.Categorie3_Name.isin(allowed_guess)]
gg = rayon.groupby('Categorie1')

print 'rayon...'

test = pd.read_csv(ddir+'test.csv',sep=';').fillna('')
add_txt(test)
test.txt = map(normalize_guess,test.txt)

print 'test...'

resultat = pd.read_csv(rname,sep=';')

df = test.merge(resultat,'left',None,'Identifiant_Produit','Id_Produit')
df = df.merge(rayon,'left',None,'Id_Categorie','Categorie3')
df.Categorie1 = map(lambda a:cat3tocat1[a],df.Id_Categorie)

print 'guess>>'
df['guess'] = map(lambda (i,r):all_guess(i,r),df.iterrows())
#df['guess'] = df.Id_Categorie
print 'guess<<'

diff = df[df.Id_Categorie != df.guess]
diff = diff[['Identifiant_Produit','Description','Libelle','Marque','prix','Categorie3_Name','guess']]
diff = diff.merge(rayon,'left',None,'guess','Categorie3')
diff = diff[[u'guess',u'Categorie3_Name_x',u'Categorie3_Name_y',  u'Description', u'Libelle', u'Marque', u'prix']]
diff.to_csv('hard_diff.csv',sep=';',index=False)

print 'correction made :',len(diff)

df.Id_Categorie = df.guess
guess = df[['Id_Produit','Id_Categorie']]
guess = guess.drop_duplicates()
guess.to_csv(ddir+'resultat.hard_guess.csv',sep=';',index=False)

