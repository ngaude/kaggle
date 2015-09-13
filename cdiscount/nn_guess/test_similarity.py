import pandas as pd
from Levenshtein import jaro
from utils import normalize_libelle,normalize_txt,normalize_guess
import matplotlib.pyplot as plt
import numpy as np

ddir = '/home/ngaude/workspace/data/cdiscount/'

test = pd.read_csv(ddir+'test.csv',sep=';').fillna('');
test['lib'] = map(normalize_guess,test.Libelle.values)
test = test.sort('lib').reset_index(drop=True)

resultat = pd.read_csv(ddir+'test.csv',sep=';').fillna('');

a = test.lib.values
b = [0]
for i in range(0,len(a)-1):
    if len(a[i])<8 or len(a[i+1])<8:
        b.append(0)
    else:
        b.append(jaro(a[i],a[i+1]))

"""
plt.hist(b,bins=300,cumulative=True)
plt.show()
"""

cut_threshold = np.percentile(b,50)
same_categorie_than_previous_item = [i>cut_threshold for i in b]

group_categorie = [0]*len(same_categorie_than_previous_item)

for i in range(1,len(same_categorie_than_previous_item)):
    if same_categorie_than_previous_item[i] == True:
        group_categorie[i] = group_categorie[i-1]
    else:
        group_categorie[i] = group_categorie[i-1]+1

test['b'] = b
test['same_categorie_than_previous_item'] = same_categorie_than_previous_item
test['group_categorie'] = group_categorie


