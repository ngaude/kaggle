import pandas as pd
from collections import Counter
from os.path import isfile
import numpy as np

#ddir = '/home/ngaude/workspace/data/cdiscount/'
ddir = '/home/ngaude/workspace/data/resultat/'


with open(ddir+'chosen2_resultat_list.txt') as f:
    a = f.read().split('\n')[:-1]

dfs = [pd.read_csv(ddir+filename,sep=';').sort('Id_Produit') for filename in a]

n = len(dfs)

c = np.zeros(shape=(n,n))

for i in range(n):
    print i,'/',n
    for j in range(n):
        c[i,j] = sum(dfs[j].Id_Categorie.values == dfs[i].Id_Categorie.values)

def group_correlation(g):
    cc = 0
    for i in range(len(g)):
        for j in range(i,len(g)):
            cc += c[g[i],g[j]]
    return cc

gmin = range(8)
ccmin = group_correlation(gmin)
while True:
    g = np.random.permutation(n)[:8]
    cc = group_correlation(g)
    if cc < ccmin:
        gmin = g[:]
        ccmin = cc
        print ccmin

print '\n'.join([a[i] for i in gmin ])

correlation = 

df = pd.concat(dfs).reset_index()
g = df.groupby('Id_Produit')

f = open(ddir+'resultat.majority.csv','w')
f.write('Id_Produit;Id_Categorie\n')
for i in g.Id_Categorie:
    pdt = i[0]
    freq = [(vote,cat) for cat,vote in Counter(i[1]).iteritems()]
    freq = sorted(freq,reverse=True)
    num_vote = sum(zip(*freq)[0])
    first_categorie = freq[0][1]
    first_confidence = float(freq[0][0])/num_vote
    if len(freq)==1:
        second_categorie = 0
        second_confidence = 0
    else:
        second_categorie = freq[1][1]
        second_confidence = float(freq[1][0])/num_vote
    #a = (first_categorie,first_confidence,second_categorie,second_confidence)
    a = (int(first_categorie),)
    s = ';'.join(map(str,a))
    f.write(str(pdt)+';'+s+'\n')

f.close()

