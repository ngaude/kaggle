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

#train_file = ddir+'training1M.tsv'

#if not os.path.isfile(train_file):
#    with open(train_file,'w') as f:
#        df = pd.read_csv(ddir + 'trainingShuffle.tsv',sep='\t',skiprows=0,nrows=100000)
#        df.to_csv(train_file,index=False,sep='\t')
#
#
#
#train_df = pd.read_csv(train_file,sep='\t')

test_file = ddir+'test.csv'
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
#train_X = vectorizer.transform(iterText(train_df))
#train_y = train_df.Categorie3

#classifier = SGDClassifier()
#classifier.fit(train_X[0:100000],train_y[0:100000],class_weight = cat_freq(train_df[0:100000]))
#print classifier.score(train_X[0:100000],train_y[0:100000])
#print classifier.score(train_X[100000:200000],train_y[100000:200000])
#print classifier.score(train_X[200000:300000],train_y[200000:300000])
# 84% on fitted train
# 72.9% on predicted train+
# 73.0% on predicted train++
# sounds good, proceed 


#train_df = pd.read_csv(ddir+'trainingShuffle.tsv',sep='\t',nrows = 4000000)
#train_X = vectorizer.transform(iterText(train_df))
#train_y = train_df.Categorie3
#
#dist=[]
#
#size_r = 100
#size_c = 3200000
#X_t = test_X[:size_r]
#X_tt = train_X[:size_r]
#X_c = train_X[size_r:size_c]
#nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute',
#                        metric='cosine').fit(X_c)
#t_distances,_ = nbrs.kneighbors(X_t)
#tt_distances,_ = nbrs.kneighbors(X_tt)
#dist.append((size_c,np.median(tt_distances),np.median(t_distances)))
#
#
##dist=[]
#size_r = 100
#size_c = 200000
#X_t = test_X[:size_r]
#X_tt = train_X[:size_r]
#X_c = train_X[size_r:size_c]
#from sklearn.neighbors import LSHForest
#lshf = LSHForest(n_estimators=16)
#lshf.fit(X_c)
#t_distances,_ = lshf.kneighbors(X_t)
#tt_distances,_ = lshf.kneighbors(X_tt)
#dist.append((size_c,np.median(tt_distances),np.median(t_distances)))

##########################################################
# build X_sample as closest X_train neighbors from test_X
##########################################################


train_df = pd.read_csv(ddir+'trainingShuffle.tsv',sep='\t',nrows = 3000000)
train_X = vectorizer.transform(iterText(train_df))
train_y = train_df.Categorie3

m = 300 # number of train slicing searching for best 
n = test_X.shape[0] # 35065
dist=np.zeros(shape=(n,m),dtype=float)
idx=np.zeros(shape=(n,m),dtype=int)

from sklearn.neighbors import NearestNeighbors

for i in range(m):
    print i,'/',m
    size_c = 10000
    off_c = i*size_c
    X_t = test_X
    X_c = train_X[off_c:off_c+size_c]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute',metric='cosine').fit(X_c)
    t_dist,t_idx = nbrs.kneighbors(X_t)
    dist[:,i] = t_dist[:,0]
    idx[:,i] = t_idx[:,0]+off_c

sorting = numpy.argsort(dist, axis=1)

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

print 'test sample size',len(set(best_idx))
print 'train2test median distance',np.median(best_dist)


#indices.shape = indices.shape[0]*indices.shape[1]
#distances.shape = distances.shape[0]*distances.shape[1]
sample_id = sorted(set(best_idx))

sample_X = train_X[sample_id,:]
sample_y = train_y[sample_id]
sample_w = cat_freq(train_df.loc[sample_id])






# train a SGD classifier on the X_sample very fitted sample of training according X_test distances
classifier = SGDClassifier()
classifier.fit(sample_X,sample_y,class_weight = sample_w)
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

def compare_resultat(f1,f2):
    df1 = pd.read_csv(f1,sep=';')
    df2 = pd.read_csv(f1,sep=';')
    cmp_score = sum(df1.Id_Categorie == df2.Id_Categorie)*1./len(df1)
    return cmp_score


submit_file = ddir+'resultat4.csv'
#test_df = pd.read_csv(test_file,sep=';')
test_df['Id_Produit']=test_df['Identifiant_Produit']
test_df['Id_Categorie'] = classifier.predict(test_X)
test_df = test_df[['Id_Produit','Id_Categorie']]
test_df.to_csv(submit_file,sep=';',index=False)

## comparison with :
## resultat1.csv scored 15,87875%
## resultat2.csv scored 20,66930%
## resultat3.csv scored 37,52794% (train2test median distance 0.481 and sample size ~39K)
## resultat4.csv scored 43,80265% (train2test median distance 0.418 and sample size 47242)

