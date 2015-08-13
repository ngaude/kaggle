import pandas as pd
from collections import Counter
import sys
from os.path import isfile

ddir = '/home/ngaude/workspace/data/cdiscount/'

assert len(sys.argv) == 3  ##### usage guess.py $RESULTAT.CSV $SEARCH_RESULTAT.TXT ####
rname = sys.argv[1]
sname = sys.argv[2]
assert isfile(ddir+rname) ##### usage guess.py $RESULTAT.CSV $SEARCH_RESULTAT.TXT ####
assert isfile(ddir+sname) ##### usage guess.py $RESULTAT.CSV $SEARCH_RESULTAT.TXT ####

nn = {}

D = []

with open(ddir+sname) as f:
    for l in f:
        if l.startswith('*****'):
            continue
        if l.startswith('Not Found.'):
            continue
        if l.startswith('query'):
            Identifiant_Produit = int(l.split(':')[1])
            continue
        nn_Identifiant_Produit = int(l.split('\t')[1].split('@')[0])
        D.append(float(l.split(':')[1]))
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
    #print Id_Produit,first_categorie,first_confidence,num_vote
    Id_Produit_list.append(Id_Produit)
    Id_Categorie_list.append(first_categorie)
    confidence_list.append(first_confidence)
    num_vote_list.append(num_vote)


d = {'Id_Produit': Id_Produit_list, 
    'Id_Categorie': Id_Categorie_list,
    'confidence' : confidence_list,
    'num_vote' : num_vote_list}

df = pd.DataFrame(data=d)
df = df[df.confidence > 0.5]

resultat = pd.read_csv(ddir+rname,sep=';')

m = df.merge(resultat,'left','Id_Produit')

same = sum(m.Id_Categorie_x == m.Id_Categorie_y)
correction = sum(m.Id_Categorie_x != m.Id_Categorie_y)

print 'index searching provides',len(m),'guesses for',correction,'correction'


d = {k:v for k,v in zip(m.Id_Produit,m.Id_Categorie_x)}
resultat.Id_Categorie = resultat.apply(lambda r:d.get(r.Id_Produit,r.Id_Categorie),axis=1)

resultat.to_csv(ddir+'resultat.index_search.csv',sep=';',index=False)
