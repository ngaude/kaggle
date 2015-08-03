import pandas as pd
import numpy as np
from utils import ddir,header,training_sample
from utils import training_sample

df = pd.read_csv(ddir+'training_shuffled_normed.csv',sep=';',names = header()).fillna('')

dftrain = df.ix[:15000000]
dfvalid = df.ix[15000000:]

a = dftrain.groupby('Categorie3').Identifiant_Produit.count().values
K3 = np.sqrt(np.mean(a)*np.median(a)) # 631

dfs = training_sample(dftrain,'Categorie3',K3,maxsampling=30)
dfv = training_sample(dfvalid,'Categorie3',10,maxsampling=1)

np.nunique(dfs.Identifiant_Produit)

