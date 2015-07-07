# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

import pandas as pd
import os.path
import string
from string import maketrans   # Required to call maketrans function.
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.linear_model import SGDClassifier
import numpy as np

os.chdir('C:/Users/ngaude/Documents/GitHub/kaggle/cdiscount/')
STOPWORDS = []
with open('stop-words_french_1_fr.txt', "r") as f:
    STOPWORDS += f.read().split('\n')
with open('stop-words_french_2_fr.txt', "r") as f:
    STOPWORDS += f.read().split('\n')
STOPWORDS = set(STOPWORDS)


fpath = 'E:/workspace/data/cdiscount/'
ifname = fpath+'training.csv'
ofname = fpath+'training.tsv'
tfname = fpath+'training4M.tsv'
sfname = fpath+'trainingShuffle.tsv'
rfname = fpath+'test.csv'
resultat = fpath+'resultat.csv'

if not os.path.isfile(ofname):
    print 'creating :',ofname
    with open(ofname,'w') as f:
        for i,l in enumerate(open(ifname)):
            if i%100000==0:
                print i
            f.write(l.translate(maketrans(';','\t')))        

if not os.path.isfile(tfname) or not os.path.isfile(sfname):
    print 'creating :',tfname
    df = pd.read_csv(ofname,sep='\t')
    df = df.reindex(np.random.permutation(df.index))
    df.to_csv(sfname,sep='\t')
    df.head(4000000).to_csv(tfname,sep='\t')

nb_txt = 50000
nb_train = 100000
nb_test = 20000

df = pd.read_csv(sfname,sep='\t',nrows = nb_txt+nb_train+nb_test)
    


intab = string.punctuation
outtab = ' '*len(intab)
trantab = string.maketrans(intab, outtab)

def text_normed(txt):
    txt = txt.translate(trantab).lower()
    txt = txt.lower()
    tokens = [w for w in txt.split(' ') if (len(w)>2) and (w not in STOPWORDS) ]
    return ' '.join(tokens)

class iterText(object):
    def __init__(self, df):
        """
        Yield each document in turn, as a text.
        """
        self.df = df
    
    def __iter__(self):
        for row_index, row in self.df.iterrows():
            if row_index%1000==0:
                print row_index
            d = m = l = ''
            if type(row.Description) is str:
                d = text_normed(row.Description)
            if type(row.Libelle) is str:
                l = text_normed(row.Libelle)
            if type(row.Marque) is str:
                m = text_normed(row.Marque)
            txt = ' '.join([m]*5+[l]*3+[d])
            yield txt
    
    def __len__(self):
        return len(self.df)
  
def cat_freq(df):
    # compute Categorie3 classe frequency
    sfreq = len(df)*1.
    g = df.groupby('Categorie3').Libelle.count()/sfreq
    return dict(g)

train = df.head(nb_train)
test = df.tail(nb_test)
txt = df[nb_train:nb_train+nb_txt]

vectorizer = TfidfVectorizer(
    min_df = 0.00009,
    max_features=nb_txt,
    stop_words=None, # 'french' is not supported so far
    smooth_idf=True,
    norm='l2',
    sublinear_tf=False,
    use_idf=True,
    ngram_range=(1,3))

vectorizer.fit(iterText(txt))

X = vectorizer.transform(iterText(train))
y = train.Categorie3
clf = SGDClassifier()
clf.fit(X,y,class_weight = cat_freq(train))
print clf.score(X, y)

Xt = vectorizer.transform(iterText(test))
yt = test.Categorie3
print clf.score(Xt, yt)

#######################
# RESULTAT SUBMISSION #
#######################

#rdf = pd.read_csv(rfname,sep=';')
#Xtest = vectorizer.transform(iterText(rdf))
#rdf['Id_Produit']=rdf['Identifiant_Produit']
#rdf['Id_Categorie'] = clf.predict(Xtest)
#rdf = rdf[['Id_Produit','Id_Categorie']]
#rdf.to_csv(resultat,sep=';',index=False)

##############
# VALIDATION #
##############
vdf = pd.read_csv(sfname,skiprows = (nb_text+nb_train+nb_txt)+1000000,nrows=50000,sep=';')
Xv = vectorizer.transform(iterText(vdf))
yv = vdf.Categorie3
print clf.score(Xv, yv)


