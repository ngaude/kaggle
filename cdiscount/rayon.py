import pandas as pd
from utils import wdir,ddir,header,normalize_txt

df = pd.read_csv(ddir+'rayon.csv',sep=';')
df['txt'] = df.Categorie1_Name+' '+df.Categorie2_Name+' '+df.Categorie3_Name
df.txt = df.apply(lambda r:normalize_txt(r.txt),axis=1)
dfrayon = df
wrayon = set((' '.join(list(dfrayon.txt))).split())

df = pd.read_csv(ddir+'training_shuffled_normed.csv',sep=';',names=header())
df['txt'] = (df.Marque+' ')*3+(df.Libelle+' ')*2+df.Description*1
dftrain = df
wtrain = set((' '.join(list(dftrain.txt))).split())

df = pd.read_csv(ddir+'test_normed.csv',sep=';',names=header(test=True))
df['txt'] = [df.Marque]*3+[df.libelle]*2+[df.Description]*1
dftest = df
wtest = set((' '.join(list(dftest.txt))).split())

df = pd.read_csv(ddir+'validation_normed.csv',sep=';',names=header())
df['txt'] = [df.Marque]*3+[df.libelle]*2+[df.Description]*1
dfvalid = df
wvalid = set((' '.join(list(dfvalid.txt))).split())

