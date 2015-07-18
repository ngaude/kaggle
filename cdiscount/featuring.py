# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

import pandas as pd
import os.path
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from bs4 import BeautifulSoup
from sklearn.linear_model import SGDClassifier
import numpy as np
import unicodedata 
import re
from sklearn.externals import joblib
import Stemmer
import matplotlib.pyplot as plt

#ddir = 'E:/workspace/data/cdiscount/'
#wdir = 'C:/Users/ngaude/Documents/GitHub/kaggle/cdiscount/'
ddir = '/home/ngaude/workspace/data/cdiscount/'
wdir = '/home/ngaude/workspace/github/kaggle/cdiscount/'

stopwords = []
with open(wdir+'stop-words_french_1_fr.txt', "r") as f:
    stopwords += f.read().split('\n')

with open(wdir+'stop-words_french_2_fr.txt', "r") as f:
    stopwords += f.read().split('\n')

stopwords += nltk.corpus.stopwords.words('french')
stopwords += ['voir', 'presentation']
stopwords = set(stopwords)
stemmer = Stemmer.Stemmer('french')

def normalize_txt(txt):
    # remove html stuff
    txt = BeautifulSoup(txt,from_encoding='utf-8').get_text()
    # lower case
    txt = txt.lower()
    # special escaping character '...'
    txt = txt.replace(u'\u2026','.')
    txt = txt.replace(u'\u00a0',' ')
    # remove accent btw
    txt = unicodedata.normalize('NFD', txt).encode('ascii', 'ignore')
    #txt = unidecode(txt)
    # remove non alphanumeric char
    txt = re.sub('[^a-z_-]', ' ', txt)
    # remove french stop words
    tokens = [w for w in txt.split() if (len(w)>2) and (w not in stopwords)]
    # french stemming
    tokens = stemmer.stemWords(tokens)
    return ' '.join(tokens)

def cat_freq(df):
    # compute Categorie3 classe frequency
    sfreq = len(df)*1.
    g = df.groupby('Categorie3').Libelle.count()/sfreq
    return dict(g)

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
                d = normalize_txt(row.Description)
            if type(row.Libelle) is str:
                l = normalize_txt(row.Libelle)
            if (type(row.Marque) is str) and (row.Marque != 'AUCUNE'):
                m = re.sub('[^a-zA-Z0-9]', '_', row.Marque).lower()
            txt = ' '.join([m]*3+[l]*2+[d])
            yield txt
    
    def __len__(self):
        return len(self.df)

test_file = ddir+'test.csv'
test_df = pd.read_csv(test_file,sep=';')

vectorizer = TfidfVectorizer(
    min_df = 0.00005,
    max_features=123456,
    stop_words=stopwords,
    strip_accents = 'unicode',
    smooth_idf=True,
    norm='l2',
    sublinear_tf=True,
    use_idf=True,
    ngram_range=(1,3))

vectorizer.fit(iterText(test_df))  

test_X = vectorizer.transform(iterText(test_df))
del test_df


joblib.dump(test_X,ddir+'joblib/test_X')
nrows = 1000000
columns = ['Identifiant_Produit','Categorie1','Categorie2','Categorie3','Description','Libelle','Marque','Produit_Cdiscount','prix'] 
train_df = pd.read_csv(ddir+'training_shuffled.csv',sep=';',nrows = nrows,names = columns)
train_X = vectorizer.transform(iterText(train_df))
train_y = train_df.Categorie3
del train_df

joblib.dump((train_X,train_y),ddir+'joblib/train_XY')


##########################################################
# build X_sample as closest X_train neighbors from test_X
##########################################################
"""

test_X = joblib.load(ddir+'joblib/test_X')
(train_X,train_y) = joblib.load(ddir+'joblib/train_XY')
"""

m = 1000 # number of train slicing searching for best 
n = test_X.shape[0] # 35065
dist=np.zeros(shape=(n,m),dtype=float)
idx=np.zeros(shape=(n,m),dtype=int)
nrows = train_X.shape[0]

from sklearn.neighbors import NearestNeighbors

assert nrows % m == 0

for i in range(m):
    print i,'/',m
    size_c = nrows/m
    off_c = i*size_c
    X_t = test_X
    X_c = train_X[off_c:off_c+size_c]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute',metric='cosine').fit(X_c)
    t_dist,t_idx = nbrs.kneighbors(X_t)
    dist[:,i] = t_dist[:,0]
    idx[:,i] = t_idx[:,0]+off_c

sorting = np.argsort(dist, axis=1)

best_dist=np.zeros(shape=(n,3),dtype=float)
best_idx=np.zeros(shape=(n,3),dtype=int)

for i in range(n):
    best_dist[i,0] = dist[i,sorting[i,0]]
    best_dist[i,1] = dist[i,sorting[i,1]]
    best_dist[i,2] = dist[i,sorting[i,2]]
    best_idx[i,0] = idx[i,sorting[i,0]]
    best_idx[i,1] = idx[i,sorting[i,1]]
    best_idx[i,2] = idx[i,sorting[i,2]]

best_idx.shape = best_idx.shape[0]*best_idx.shape[1]

plt.hist(np.mean(best_dist,axis=1),bins=100)
plt.show(block=False)

print 'test sample size',len(set(best_idx))
print 'train2test median distance',np.median(best_dist)


#indices.shape = indices.shape[0]*indices.shape[1]
#distances.shape = distances.shape[0]*distances.shape[1]
sample_id = sorted(set(best_idx))

sample_X = train_X[sample_id,:]
sample_y = train_y[sample_id]
sample_w = cat_freq(train_df.loc[sample_id])

joblib.dump((sample_X,sample_y,sample_w),ddir+'joblib/sampleXYW')

##########################################################
# train a SGD classifier on X_sample 
# that are made of  training samples 
# very fitted according to X_test distances
##########################################################

(sample_X,sample_y,sample_w) = joblib.load(ddir+'joblib/sampleXYW')

classifier = SGDClassifier()
classifier.fit(sample_X,sample_y,class_weight = sample_w)
classifier.sparsify()

joblib.dump(classifier,ddir+'joblib/classifier')

##########################################################

classifier = joblib.load(ddir+'joblib/classifier')

print classifier.score(sample_X,sample_y)
print classifier.score(train_X[100000:100000+n],train_y[100000:100000+n])
print classifier.score(train_X[200000:200000+n],train_y[200000:100000+n])

# 85.5% on fitted sample
# 73.12% on predicted train+
# 73.17% on predicted train++
# fingers crossed,hope this reflect the submission...
#sample_score = 0.7315

########################
## RESULTAT SUBMISSION #
########################

submit_file = ddir+'resultat14.csv'
test_file = ddir+'test.csv'
test_df = pd.read_csv(test_file,sep=';')
test_X = joblib.load(ddir+'joblib/test_X')
test_df['Id_Produit']=test_df['Identifiant_Produit']
test_df['Id_Categorie'] = classifier.predict(test_X)
test_df = test_df[['Id_Produit','Id_Categorie']]
test_df.to_csv(submit_file,sep=';',index=False)

## comparison with :
## resultat1.csv scored 15,87875%
## resultat2.csv scored 20,66930%
## resultat3.csv scored 37,52794% (train2test median distance 0.481 and sample size ~39K)
## resultat4.csv scored 43,80265% (train2test median distance 0.418 and sample size 47242)

## resultat13.csv ... 39,46479%....
## resultat14.csv ... 40,28995%.... + stemming/removal ,....

