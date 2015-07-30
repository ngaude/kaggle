import pandas as pd
from collections import Counter

ddir = '/home/ngaude/workspace/data/cdiscount/'
wdir = '/home/ngaude/workspace/github/kaggle/cdiscount/'


dfs = [pd.read_csv(ddir+'resultat'+str(i)+'.csv',sep=';') for i in [29,31,33,35,36]]
df = pd.concat(dfs).reset_index()
g = df.groupby('Id_Produit')

f = open(ddir+'resultat_majority.csv','w')
f.write('Id_Produit;Id_Categorie\n')
for i in g.Id_Categorie:
    pdt = i[0]
    freq = {v:k for k,v in Counter(i[1]).iteritems()}
    cat = freq[max(freq)]
    f.write(str(pdt)+';'+str(cat)+'\n')

f.close()

