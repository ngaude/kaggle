import pandas as pd
import numpy as np
from sklearn.externals import joblib
from utils import header
from collections import Counter

ddir = '/home/ngaude/workspace/data/cdiscount/'

(Dneighbor,IDneighbor) = joblib.load(ddir+'joblib/DIDneighbor')


dftrain = pd.read_csv(ddir+'training.csv',sep=';').fillna('')

IDcat = {}
for i,r in dftrain.iterrows():
    if i%10000 == 0:
        print i
    IDcat[r.Identifiant_Produit] = r.Categorie3


K=5

print 'median 5NN distance is ',np.median(Dneighbor[:,0:K])


df = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')

def vote_and_confidence(IDnn):
    Id_Categorie_list = []
    confidence_list = []
    for i in range(IDnn.shape[0]):
        freq = [(vote,cat) for cat,vote in Counter(IDnn[i,:K]).iteritems()]
        freq = sorted(freq,reverse=True)
        first_categorie = freq[0][1]
        first_confidence = float(freq[0][0])/K
        Id_Categorie_list.append(first_categorie)
        confidence_list.append(first_confidence)
    return (Id_Categorie_list,confidence_list)

IDnn = np.zeros(shape=(len(df),K))

for i in range(IDnn.shape[0]):
    for j in range(IDnn.shape[1]):
        IDnn[i,j] = IDcat[IDneighbor[i,j]]

(cat,conf) = vote_and_confidence(IDnn)

df['Id_Produit'] = df.Identifiant_Produit
df['Id_Categorie'] = cat
df['confidence'] = conf
df['D'] = np.median(Dneighbor,axis=1)

df = df[['Id_Produit','Id_Categorie','confidence','D']]

df.to_csv(ddir+'confidence.nn.5NN.test.csv',sep=';',index=False)
