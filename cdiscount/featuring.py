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

import time
from sklearn.neighbors import NearestNeighbors


(Xtrain,Ytrain) = joblib.load(ddir+'joblib/train_XY')
Xtest = joblib.load(ddir+'joblib/test_X')


test_count = Xtest.shape[0]
train_count = Xtrain.shape[0]

neighbors = [[] for i in range(test_count)]

def neighbor_select(test_id,dist,indx):
    if len(neighbors[test_id])>100:
        neighbors[test_id].sort()
        neighbors[test_id] = neighbors[test_id][:50]
    neighbors[test_id]+= zip(dist,indx)

def neighbor_distance(k):
    return np.median([ zip(*tup[:k])[0] if tup else [1]*k for tup in neighbors])

batch_size = 1000
k = 5
start_time = time.time()
for i in range(0,train_count,batch_size):
    if (i/batch_size)%10==0:
        print 'neighbor:',i,'/',train_count,'median distance=',neighbor_distance(k),'time=',int(time.time() - start_time),'s'
    Xb = Xtrain[i:i+min(batch_size,train_count-i)]
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute',metric='cosine').fit(Xb)
    dist,indx = nbrs.kneighbors(Xtest)
    for j in range(0,test_count):
        neighbor_select(j,dist[j,:],indx[j,:]+i)

Ineighbor = np.zeros(shape=(test_count,50),dtype = int)
Dneighbor = np.zeros(shape=(test_count,50),dtype = float)

for i in range(test_count):
    neighbors[i].sort()
    Dneighbor[i,:] = zip(*neighbors[i])[0][:50]
    Ineighbor[i,:] = zip(*neighbors[i])[1][:50]

#save raw list of the top 50 at least neighbors
joblib.dump((Dneighbor,Ineighbor),ddir+'joblib/DIneighbor')

# select for each test the 5-closest neighbors
neighbors_indices = sorted(set(Ineighbor[:,:k].flatten()))


# save neighbors
Yneighbor = Ytrain[neighbors_indices]
Xneighbor = Xtrain[neighbors_indices]

nrows = train_count
columns = ['Identifiant_Produit','Categorie1','Categorie2','Categorie3','Description','Libelle','Marque','Produit_Cdiscount','prix'] 
train_df = pd.read_csv(ddir+'training_shuffled.csv',sep=';',nrows = nrows,names = columns)

sample_X = Xtrain[neighbors_indices,:]
sample_y = Ytrain[neighbors_indices]
sample_w = cat_freq(train_df.loc[neighbors_indices])
del Ytrain
del Xtrain

joblib.dump((sample_X,sample_y,sample_w),ddir+'joblib/sampleXYW')

"""
columns = ['Identifiant_Produit','Categorie1','Categorie2','Categorie3','Description','Libelle','Marque','Produit_Cdiscount','prix'] 
train_df = pd.read_csv(ddir+'training_shuffled_tail.csv',sep=';',names = columns)
(Dneighbor,Ineighbor) = joblib.load(ddir+'joblib/DIvalidation')
k=2
neighbors_indices = sorted(set(Ineighbor[:,:k].flatten()))
valid_df = train_df.loc[neighbors_indices]
valid_X = vectorizer.transform(iterText(valid_df))
valid_Y = valid_df.Categorie3
classifier = joblib.load(ddir+'joblib/classifier')
classifier.score(valid_X,valid_Y)
# k=5 : 59%
# k=2 : 58.5%
# k=1 : 57.5%
"""

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

submit_file = ddir+'resultat15.csv'
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
## resultat14.csv ... 40,28995%.... + stemming/removal (k=3),....
## resultat15.csv ... 38,11243%.... + exact k-nn (k=5) ,....

