import pandas as pd
from collections import Counter
from os.path import isfile

#ddir = '/home/ngaude/workspace/data/cdiscount/'
ddir = '/home/ngaude/workspace/data/resultat/'

nn_filenames = ['resultat.nn.0.csv',
    'resultat.nn.1.csv',
    'resultat.nn.2.csv',
    'resultat.nn.4.csv',
    'resultat.nn.5.csv',
    'resultat.nn.6.csv',
    'resultat.nn.8.csv',
    'resultat.nn.9.csv',
    'resultat.nn.10.csv',
    'resultat.nn.12.csv',
    'resultat.nn.13.csv',
    'resultat.nn.14.csv',
    'resultat.nn.15.csv',
    'resultat.nn.16.csv',
    'resultat.nn.17.csv',
    'resultat.nn.18.csv',
    'resultat.nn.19.csv',
    'resultat.nn.20.csv',
    'resultat.nn.100.csv',
    'resultat.nn.101.csv',
    'resultat.nn.102.csv',
    'resultat.nn.103.csv',
    'resultat.nn.104.csv',
    'resultat.nn.105.csv',
    'resultat.nn.106.csv',
    'resultat.nn.107.csv',
    'resultat.nn.108.csv',
    'resultat.nn.109.csv',
    'resultat.nn.110.csv',
    'resultat.nn.111.csv',
    'resultat.nn.112.csv',
    'resultat.nn.113.csv',
    'resultat.nn.114.csv',
    'resultat.nn.115.csv',
    'resultat.nn.116.csv']

auto_filenames = ['resultat.auto.100.csv',
    'resultat.auto.101.csv',
    'resultat.auto.102.csv',
    'resultat.auto.103.csv',
    'resultat.auto.104.csv',
    'resultat.auto.105.csv',
    'resultat.auto.106.csv',
    'resultat.auto.107.csv',
    'resultat.auto.108.csv',
    'resultat.auto.109.csv',
    'resultat.auto.110.csv',
    'resultat.auto.111.csv',
    'resultat.auto.112.csv',
    'resultat.auto.113.csv',
    'resultat.auto.114.csv',
    'resultat.auto.115.csv',
    'resultat.auto.116.csv',
    'resultat.auto.200.csv',
    'resultat.auto.201.csv',
    'resultat.auto.202.csv',
    'resultat.auto.203.csv',
    'resultat.auto.204.csv',
    'resultat.auto.205.csv',
    'resultat.auto.206.csv',
    'resultat.auto.207.csv',
    'resultat.auto.208.csv',
    'resultat.auto.209.csv',
    'resultat.auto.210.csv',
    'resultat.auto.211.csv',
    'resultat.auto.212.csv',
    'resultat.auto.213.csv',
    'resultat.auto.214.csv',
    'resultat.auto.215.csv',
    'resultat.auto.216.csv',
    'resultat.auto.217.csv',
    'resultat.auto.218.csv',
    'resultat.auto.219.csv',
    'resultat.auto.220.csv',
    'resultat.auto.300.csv',
    'resultat.auto.301.csv',
    'resultat.auto.302.csv',
    'resultat.auto.303.csv',
    'resultat.auto.304.csv',
    'resultat.auto.305.csv',
    'resultat.auto.306.csv',
    'resultat.auto.307.csv',
    'resultat.auto.308.csv',
    'resultat.auto.309.csv',
    'resultat.auto.310.csv',
    'resultat.auto.311.csv']

dfs = [pd.read_csv(ddir+filename,sep=';') for filename in nn_filenames]

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
    a = (first_categorie,)
    s = ';'.join(map(str,a))
    f.write(str(pdt)+';'+s+'\n')

f.close()

