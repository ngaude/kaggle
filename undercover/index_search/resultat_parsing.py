import pandas as pd
from collections import Counter

ddir = '/home/ngaude/workspace/data/cdiscount/'
nn = {}

with open(ddir+'index_search_resultat.txt') as f:
    for l in f:
        if l.startswith('*****'):
            continue
        if l.startswith('Not Found.'):
            continue
        if l.startswith('query'):
            Identifiant_Produit = int(l.split(':')[1])
            continue
        nn_Identifiant_Produit = int(l.split('\t')[1])
        nn.setdefault(Identifiant_Produit,[]).append(nn_Identifiant_Produit)

Id_Produit_list = []
Id_Categorie_list = []
confidence_list = []
num_vote_list = []

for Id_Produit,nn_list in nn.iteritems():
    freq = [(vote,cat) for cat,vote in Counter(nn_list).iteritems()]
    freq = sorted(freq,reverse=True)
    num_vote = sum(zip(*freq)[0])
    first_categorie = freq[0][1]
    first_confidence = float(freq[0][0])/num_vote
    print Id_Produit,first_categorie,first_confidence,num_vote
    Id_Produit_list.append(Id_Produit)
    Id_Categorie_list.append(first_categorie)
    confidence_list.append(first_confidence)
    num_vote_list.append(num_vote)


d = {'Id_Produit': Id_Produit_list, 
    'Id_Categorie': Id_Categorie_list,
    'confidence' : confidence_list,
    'num_vote' : num_vote_list}

df = pd.DataFrame(data=d)

resultat = pd.read_csv(ddir+'proba.auto.merging.70.csv',sep=';')
resultat = pd.read_csv(ddir+'resultat.merge70.hard.nn.csv',sep=';')

m = df.merge(resultat,'left','Id_Produit')

diff = m[m.Id_Categorie_x != m.Id_Categorie_y]
diff = m[(m.Id_Categorie_x != m.Id_Categorie_y) & (m.num_vote>1) & (m.confidence > 0.5)]





