# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

import pandas as pd
import os.path
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.linear_model import SGDClassifier
import numpy as np

# data & working directories
ddir = 'E:/workspace/data/cdiscount/'
wdir = 'C:/Users/ngaude/Documents/GitHub/kaggle/cdiscount/'

os.chdir(wdir)

# load french stop words list
STOPWORDS = []
with open('stop-words_french_1_fr.txt', "r") as f:
    STOPWORDS += f.read().split('\n')
with open('stop-words_french_2_fr.txt', "r") as f:
    STOPWORDS += f.read().split('\n')
STOPWORDS = set(STOPWORDS)

def cat_freq(df):
    # compute Categorie3 classe frequency
    sfreq = len(df)*1.
    g = df.groupby('Categorie3').Libelle.count()/sfreq
    return dict(g)

#from string import maketrans
#intab = string.punctuation
#outtab = ' '*len(intab)
#trantab = string.maketrans(intab, outtab)

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
                d = row.Description
            if type(row.Libelle) is str:
                l = row.Libelle
            if type(row.Marque) is str:
                m = row.Marque
            txt = ' '.join([m]*3+[l]*2+[d])
            yield txt
    
    def __len__(self):
        return len(self.df)

train_file = ddir+'training1M.tsv'
test_file = ddir+'test.csv'

if not os.path.isfile(train_file):
    with open(train_file,'w') as f:
        df = pd.read_csv(ddir + 'trainingShuffle.tsv',sep='\t',skiprows=0,nrows=1000000)
        df.to_csv(train_file,index=False,sep='\t')

train_df = pd.read_csv(train_file,sep='\t')
test_df = pd.read_csv(test_file,sep=';')

vectorizer = TfidfVectorizer(
    min_df = 0.00005,
    max_features=123456,
    stop_words=STOPWORDS,
    strip_accents = 'unicode',
    smooth_idf=True,
    norm='l2',
    sublinear_tf=False,
    use_idf=True,
    ngram_range=(1,3))

vectorizer.fit(iterText(test_df))  

test_X = vectorizer.transform(iterText(test_df))
train_X = vectorizer.transform(iterText(train_df))
train_y = train_df.Categorie3

#classifier = SGDClassifier()
#classifier.fit(train_X[0:100000],train_y[0:100000],class_weight = cat_freq(train_df[0:100000]))
#print classifier.score(train_X[0:100000],train_y[0:100000])
#print classifier.score(train_X[100000:200000],train_y[100000:200000])
#print classifier.score(train_X[200000:300000],train_y[200000:300000])
# 84% on fitted train
# 72.9% on predicted train+
# 73.0% on predicted train++
# sounds good, proceed 

from sklearn.neighbors import LSHForest
lshf = LSHForest(n_estimators=17)
lshf.fit(train_X)

distances, indices = lshf.kneighbors(test_X, n_neighbors=10)

# build X_sample as closest X_train neighbors from test_X 
indices.shape = indices.shape[0]*indices.shape[1]
distances.shape = distances.shape[0]*distances.shape[1]
sample_id = sorted(set(indices))
sample_X = train_X[sample_id,:]
sample_y = train_y[sample_id]
sample_w = cat_freq(train_df.loc[sample_id])
sample_d = distances[sample_id]

# train a SGD classifier on the X_sample very fitted sample of training according X_test distances
classifier = SGDClassifier()
classifier.fit(sample_X,sample_y,class_weight = sample_w)
print classifier.score(sample_X,sample_y)
print classifier.score(train_X[100000:200000],train_y[100000:200000])
print classifier.score(train_X[200000:300000],train_y[200000:300000])

# 85.5% on fitted sample
# 73.12% on predicted train+
# 73.17% on predicted train++
# fingers crossed,hope this reflect the submission...
#sample_score = 0.7315

########################
## RESULTAT SUBMISSION #
########################

submit_file = ddir+'resultat.csv'
#test_df = pd.read_csv(test_file,sep=';')
test_df['Id_Produit']=test_df['Identifiant_Produit']
test_df['Id_Categorie'] = classifier.predict(test_X)
test_df = test_df[['Id_Produit','Id_Categorie']]
test_df.to_csv(submit_file,sep=';',index=False)

## comparison with :
## resultat1.csv that scored 15,87875%
#submit1_file = ddir+'resultat1.csv'
#submit1_score = 0.1587875
#test1_df = pd.read_csv(submit1_file,sep=';')
#same_score = sum(test_df.Id_Categorie == test1_df.Id_Categorie)*1./len(test_df)
## upper bound estimation of what score should be if X_sample is close enough from X_test
#test_score = (1-same_score)*sample_score+submit1_score

## resultat2.csv that scored 20,66930%