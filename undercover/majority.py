import pandas as pd
from collections import Counter
from os.path import isfile

ddir = '/home/ngaude/workspace/data/cdiscount/'
ddir = '/home/ngaude/workspace/data/resultat/'


# dfs = [pd.read_csv(ddir+'resultat'+str(i)+'.csv',sep=';') for i in [53,51,49,47,45,44,36,39]]

dfs = [pd.read_csv(ddir+'resultat'+str(i)+'.csv',sep=';') for i in range(1,61) if isfile(ddir+'resultat'+str(i)+'.csv')]

df = pd.concat(dfs).reset_index()
g = df.groupby('Id_Produit')

f = open(ddir+'resultat_majority.csv','w')
f.write('Id_Produit;Id_Categorie\n')
for i in g.Id_Categorie:
    pdt = i[0]
    freq = {v:k for k,v in Counter(i[1]).iteritems()}
    cat = freq[max(freq)]
    f.write(str(pdt)+';'+str(int(cat))+'\n')

f.close()

