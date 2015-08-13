import pandas as pd
import numpy as np

ddir = '/home/ngaude/workspace/data/cdiscount/'

proba = pd.read_csv(ddir+'proba.auto.merging.70.csv',sep=';')

weak_proba = np.percentile(proba.Proba_Categorie3,10)

df = proba[proba.Proba_Categorie3 < weak_proba]

test = pd.read_csv(ddir+'test_normed.csv',sep=';',names=['Identifiant_Produit','Description','Libelle','Marque','prix']).fillna('')
test['t_index'] = test.Marque+' '+test.Libelle+' '+test.Description

rayon = pd.read_csv(ddir+'rayon.csv',sep=';')

df = df.merge(test,'left',None,'Id_Produit','Identifiant_Produit')
df = df.merge(rayon,'left',None,'Id_Categorie','Categorie3')

df.to_csv('/tmp/weak.csv',sep=';',index=False)
df = df[['t_index','Identifiant_Produit']].sort('t_index').reset_index(drop=True)


df.to_csv(ddir+'weak_txt.tsv',sep=';',index=False,header=False)
