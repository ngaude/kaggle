#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from utils import wdir,ddir,header,normalize_txt,add_txt
from sklearn.externals import joblib

from utils import itocat1,itocat2,itocat3
from utils import cat1toi,cat2toi,cat3toi
from utils import cat3tocat2,cat3tocat1,cat2tocat1
from utils import cat1count,cat2count,cat3count


dftrain = pd.read_csv(ddir+'training_shuffled_normed.csv',sep=';',names = header()).fillna('')

gg = dftrain.groupby('Marque')
cc = gg.Categorie1.unique()
ez_cat1 = {cc.index[i]:a[0] for (i,a) in enumerate(cc) if len(a)==1}
cc = gg.Categorie2.unique()
ez_cat2 = {cc.index[i]:a[0] for (i,a) in enumerate(cc) if len(a)==1}
cc = gg.Categorie3.unique()
ez_cat3 = {cc.index[i]:a[0] for (i,a) in enumerate(cc) if len(a)==1}

joblib.dump((ez_cat1,ez_cat2,ez_cat3), ddir+'/joblib/ez_cat')


##################################
# predicting from log_proba 1,2,3
##################################

(stage1_log_proba_valid,stage2_log_proba_valid,stage3_log_proba_valid) = joblib.load(ddir+'/joblib/backup/log_proba_valid')
(stage1_log_proba_test,stage2_log_proba_test,stage3_log_proba_test) = joblib.load(ddir+'/joblib/backup/log_proba_test')

(ez_cat1,ez_cat2,ez_cat3) = joblib.load(ddir+'/joblib/ez_cat')

dfvalid = pd.read_csv(ddir+'validation_perfect.csv',sep=';',names = header()).fillna('')
dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')
dfresultat = pd.read_csv(ddir+'resultat43.csv',sep=';').fillna('')

rows = np.nonzero(dfvalid.Marque.isin(ez_cat3))[0]
df = dfvalid.ix[rows]
cat3 = map(lambda m:ez_cat3[m],dfvalid.Marque[rows])
df['guess'] = cat3



rows = np.nonzero(dftest.Marque.isin(ez_cat3))[0]
cat3 = map(lambda m:ez_cat3[m],dftest.Marque[rows])

print 'easying',sum(cat3 != dfresultat.Id_Categorie[rows]),' categorie3'

# simply patch the resultat to fix the easy categorie:
dfresultat.ix[rows,'Id_Categorie'] = cat3

assert sum(cat3 != dfresultat.Id_Categorie[rows]) == 0

dfresultat.to_csv(ddir+'ez_cat.csv',sep=';',index=False)

diff = dfresultat.iloc[rows]
test = pd.read_csv(ddir+'test.csv',sep=';').fillna('')
rayon = pd.read_csv(ddir+'rayon.csv',sep=';').fillna('ZZZ')
diff = diff.merge(test,'left',None,'Id_Produit','Identifiant_Produit')
diff = diff.merge(rayon,'left',None,'Id_Categorie','Categorie3')
diff = diff[[u'Categorie3_Name',u'Description', u'Libelle', u'Marque', u'prix']]
diff.to_csv(ddir+'diff.csv',sep=';',index=False)


def result_diffing(fx,fy):
    dfx = pd.read_csv(fx,sep=';').fillna('')
    dfy = pd.read_csv(fy,sep=';').fillna('')
    test = pd.read_csv(ddir+'test.csv',sep=';').fillna('')
    rayon = pd.read_csv(ddir+'rayon.csv',sep=';').fillna('')
    dfx = dfx.merge(rayon,'left',None,'Id_Categorie','Categorie3')
    dfy = dfy.merge(rayon,'left',None,'Id_Categorie','Categorie3')
    dfx = dfx.merge(test,'left',None,'Id_Produit','Identifiant_Produit')
    df = dfx.merge(dfy,'inner','Id_Produit')
    #df = df[df.Categorie3_x != df.Categorie3_y]
    #df = df[['Id_Produit','Categorie3_Name_x','Categorie3_Name_y','Marque','Libelle','Description','prix']]
    df = df[df.Categorie1_x != df.Categorie1_y]
    df = df[['Id_Produit','Categorie1_Name_x','Categorie1_Name_y','Marque','Libelle','Description','prix']]
    df.to_csv(ddir+'diff.csv',sep=';',index=False)
    return df

fx = ddir+'resultat43.csv'
fy = ddir+'ez_cat.csv'

a = result_diffing(fx,fy)


    df = [[]]
    return 


    .reto_csv(ddir+'ez_cat.csv',sep=';',index=False)
    dfy.t






