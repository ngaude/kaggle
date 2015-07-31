#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.externals import joblib
from sklearn.neighbors import NearestNeighbors
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from os.path import basename
import random
from sklearn.externals import joblib
from joblib import Parallel, delayed


from utils import wdir,ddir,header
from utils import itocat1,itocat2,itocat3
from utils import cat1toi,cat2toi,cat3toi
from utils import cat3tocat2,cat3tocat1,cat2tocat1
from utils import cat1count,cat2count,cat3count

def add_txt(df):
    assert 'Marque' in df.columns
    assert 'Libelle' in df.columns
    assert 'Description' in df.columns
    assert 'prix' in df.columns
    df['txt'] = ('px'+(df.prix).astype(int).astype(str)+' ')+(df.Marque+' ')*3+(df.Libelle+' ')*2+df.Description
    return

def vectorizer(txt):
    vec = TfidfVectorizer(
        min_df = 2,
        stop_words = None,
        max_features=234567,
        smooth_idf=True,
        norm='l2',
        sublinear_tf=True,
        use_idf=True,
        ngram_range=(1,2))
    X = vec.fit_transform(txt)
    return (vec,X)

##################
# VECTORIZING
##################

# vectorize dftest
dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')
add_txt(dftest)
vec,Xtest = vectorizer(dftest.txt)

# vectorize dftrain
dftrain = pd.read_csv(ddir+'training_shuffled_normed.csv',sep=';',names = header()).fillna('')
add_txt(dftrain)
Ytrain = dftrain.Categorie3.values.copy()
IDtrain = dftrain.Identifiant_Produit.values.copy()


# NOTE : memory error work around... 

# let's serialize.

joblib.dump((vec,IDtrain,Ytrain),'/tmp/vecIDYtrain')
joblib.dump(Xtest,ddir+'joblib/Xtest')
with open('/tmp/train.txt','w') as f:
    for l in dftrain.txt:
        f.write(l+'\n')

#free memory
del Xtest,dftest
del vec,Ytrain,IDtrain,dftrain

# let's unserialize
(vec,IDtrain,Ytrain)=joblib.load('/tmp/vecIDYtrain')
Xtest = joblib.load(ddir+'joblib/Xtest')
class iterTxt():
    def __init__(self):
        self.fname = '/tmp/train.txt'
    def __iter__(self):
        start_time = time.time()
        for i,line in enumerate(open(self.fname)):
            if i%10000==0:
                print self.fname,': lines=',i,'time=',int(time.time() - start_time),'s'
            yield line

Xtrain = vec.transform(iterTxt())

joblib.dump((vec,IDtrain,Xtrain,Ytrain),ddir+'joblib/vecIDXYtrain')

##################
# NEIGHBORING
##################

# (vec,IDtrain,Xtrain,Ytrain) = joblib.load(ddir+'joblib/vecIDXYtrain')
# Xtest = joblib.load(ddir+'joblib/Xtest')

test_count = Xtest.shape[0]
train_count = Xtrain.shape[0]

import gc
sum([sys.getsizeof(i) for i in gc.get_objects()])

def neighbor_select(test_id,dist,indx):
    if len(neighbors[test_id])>400:
        neighbors[test_id].sort()
        prev =  neighbors[test_id]
        neighbors[test_id] = list(neighbors[test_id][:neighbor_size])
        del prev
    neighbors[test_id]+= zip(dist,indx)

def neighbor_distance(k):
    return np.median([ zip(*tup[:k])[0] if tup else [1]*k for tup in neighbors])

neighbors = [[] for i in range(test_count)]
batch_size = 10000
k = 5
neighbor_size = 200
start_time = time.time()
for i in range(0,train_count,batch_size):
    if (i/batch_size)%1==0:
        print 'neighbor:',i,'/',train_count,'median distance=',neighbor_distance(k),'time=',int(time.time() - start_time),'s'
    Xb = Xtrain[i:i+min(batch_size,train_count-i)]
    knn = NearestNeighbors(n_neighbors=k, algorithm='brute',metric='cosine').fit(Xb)
    dist,indx = knn.kneighbors(Xtest)
    for j in range(0,test_count):
        neighbor_select(j,dist[j,:],indx[j,:]+i)
    del Xb,knn

IDneighbor = np.zeros(shape=(test_count,neighbor_size),dtype = int)
Dneighbor = np.zeros(shape=(test_count,neighbor_size),dtype = float)

for i in range(test_count):
    neighbors[i].sort()
    Dneighbor[i,:] = zip(*neighbors[i])[0][:neighbor_size]
    IDneighbor[i,:] = [IDtrain[a] for a in zip(*neighbors[i])[1][:neighbor_size]]

#save a raw list of the top neighbor_size neighbors for each test sample
joblib.dump((Dneighbor,IDneighbor),ddir+'joblib/DIDneighbor')

ids = []
for l in neighbors:
    ids += zip(*l[:neighbor_size])[1]


##################
# NN  = PERFECT VALIDATION SET
##################

# select for each test sample 2-neighbors (from the top 1-11's ones).
valid_ids = []
all_ids = []
for i in range(IDneighbor.shape[0]):
    all_ids += list(IDneighbor[i,:])
    valid_ids += random.sample(IDneighbor[i,1:11],2)

all_ids = set(all_ids)
valid_ids = set(valid_ids)

dftrain = pd.read_csv(ddir+'training_shuffled_normed.csv',sep=';',names = header()).fillna('')
valid_rows = dftrain.Identifiant_Produit.isin(valid_ids)
dfvalid = dftrain[valid_rows]
dfvalid.to_csv(ddir+'validation_perfect.csv',sep=';',index=False,header=False)

##################
# NN + CLASS RATIO = PERFECT SAMPLE SET
##################

class_ratio = joblib.load(ddir+'joblib/class_ratio')
(Dneighbor,IDneighbor) = joblib.load(ddir+'joblib/DIDneighbor')

dftrain = pd.read_csv(ddir+'training_shuffled_normed.csv',sep=';',names = header()).fillna('')
dfvalid = pd.read_csv(ddir+'validation_perfect.csv',sep=';',names = header()).fillna('')
valid_ids = dfvalid.Identifiant_Produit.values

# then sample from training set according to the class_ratio ratio
# preferably from the closest neighbors

#flatten IDneighbor to get the set of all_ids that are close from test set
all_ids = []
for i in range(IDneighbor.shape[0]):
    all_ids += list(IDneighbor[i,:])
#remove valid_ids from all_ids
valid_ids = set(dfvalid.Identifiant_Produit.values)
all_ids = set(all_ids)
all_ids -= valid_ids
# convert all_ids to a suitable dataframe from training set:
all_rows = dftrain.Identifiant_Produit.isin(all_ids)
dftrain = dftrain[all_rows]

def training_sample_perfect(dftrain,N = 200,class_ratio = dict()):
    cl = dftrain['Categorie3']
    cc = cl.groupby(cl)
    s = (cc.count() > 3)
    labelmaj = s[s].index
    print len(labelmaj),'classes with ~ ',N,' samples'
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    dfs = []
    for i,cat in enumerate(labelmaj):
        if i%10==0:
            print i,'/',len(labelmaj),':'
        df = dftrain[dftrain[label] == cat]
        mincount = N*class_ratio.get(cat,1)
        if len(df)>=mincount:
            # undersample mincount samples
            rows = random.sample(df.index, mincount)
            dfs.append(df.ix[rows])
        else:
            # sample all samples + oversample the remaining
            dfs.append(df)
            df = df.iloc[np.random.randint(0, len(df), size=mincount-len(df))]
            dfs.append(df)
    dfsample = pd.concat(dfs)
    dfsample = dfsample.reset_index(drop=True)
    dfsample = dfsample.reindex(np.random.permutation(dfsample.index),copy=False)
    return dfsample

dftrain[~valid_rows]

