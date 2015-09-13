import pandas as pd
from collections import Counter
from os.path import isfile

#ddir = '/home/ngaude/workspace/data/cdiscount/'
ddir = '/home/ngaude/workspace/data/resultat/'


with open(ddir+'last_of_us.txt') as f:
    a = f.read().split('\n')[:-1]

dfs = [pd.read_csv(ddir+filename,sep=';').sort('Id_Produit') for filename in a]

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

