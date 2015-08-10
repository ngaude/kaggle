import pandas as pd
from collections import Counter
from os.path import isfile




files = ['/home/ngaude/workspace/data/resultat/proba.auto.0.csv',
    '/home/ngaude/workspace/data/resultat/proba.auto.1.csv',
    '/home/ngaude/workspace/data/resultat/proba.auto.2.csv',
    '/home/ngaude/workspace/data/resultat/proba.auto.3.csv',
    '/home/ngaude/workspace/data/resultat/proba.auto.4.csv',
    '/home/ngaude/workspace/data/resultat/proba.auto.5.csv',
    '/home/ngaude/workspace/data/resultat/proba.auto.6.csv',
    '/home/ngaude/workspace/data/resultat/proba.auto.7.csv',
    '/home/ngaude/workspace/data/resultat/proba.auto.100.csv',
    '/home/ngaude/workspace/data/resultat/proba.auto.101.csv',
    '/home/ngaude/workspace/data/resultat/proba.auto.102.csv',
    '/home/ngaude/workspace/data/resultat/proba.auto.103.csv',
    '/home/ngaude/workspace/data/resultat/proba.auto.104.csv',
    '/home/ngaude/workspace/data/resultat/proba.auto.200.csv',
    '/home/ngaude/workspace/data/resultat/proba.auto.300.csv']

dfs = [pd.read_csv(i,sep=';') for i in files]

df = pd.concat(dfs).reset_index()
g = df.groupby('Id_Produit')

f = open('resultat.majority.csv','w')
f.write('Id_Produit;Id_Categorie\n')
for i in g.Id_Categorie:
    pdt = i[0]
    freq = {v:k for k,v in Counter(i[1]).iteritems()}
    cat = freq[max(freq)]
    f.write(str(pdt)+';'+str(int(cat))+'\n')

f.close()

