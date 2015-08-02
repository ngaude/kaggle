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


from utils import wdir,ddir,header,add_txt
from utils import itocat1,itocat2,itocat3
from utils import cat1toi,cat2toi,cat3toi
from utils import cat3tocat2,cat3tocat1,cat2tocat1
from utils import cat1count,cat2count,cat3count

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

def training_sample_perfect(dftrain,N = 200,class_ratio = dict()):
    assert 'D' in dftrain.columns
    cl = dftrain.Categorie3
    cc = cl.groupby(cl)
    s = (cc.count() >= 3)
    labelmaj = s[s].index
    print 'sampling ~',N,'samples for any of',len(labelmaj),'classes'
    dfs = []
    for i,cat in enumerate(labelmaj):
        if i%10==0:
            print i,'/',len(labelmaj),':'
        df = dftrain[dftrain.Categorie3 == cat]
        df = df.sort('D',ascending=True)
        sample_count = int(np.round(N*class_ratio.get(cat,1)))
        if len(df)>=sample_count:
            # undersample sample_count samples : take the closest first
            dfs.append(df[:sample_count])
        else:
            # sample all samples + oversample the remaining
            dfs.append(df)
            df = df.iloc[np.random.randint(0, len(df), size=sample_count-len(df))]
            dfs.append(df)
    dfsample = pd.concat(dfs)
    dfsample = dfsample.reset_index(drop=True)
    dfsample = dfsample.reindex(np.random.permutation(dfsample.index),copy=False)
    return dfsample

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

import gc,sys
sum([sys.getsizeof(i) for i in gc.get_objects()])

def neighbor_select(ngh,test_id,dist,indx,nsize):
    if len(ngh[test_id])>nsize*2:
        ngh[test_id].sort()
        prev =  ngh[test_id]
        ngh[test_id] = list(ngh[test_id][:nsize])
        del prev
    ngh[test_id]+= zip(dist,indx)

def neighbor_median(ngh,k):
    return np.median([ zip(*tup[:k])[0] if tup else [1]*k for tup in ngh])

def neighbor_distance(X,Xtest,offset=0,neighbor_size=200):
    n = X.shape[0]
    lneighbors = [[] for i in range(Xtest.shape[0])]
    batch_size = 10000
    k = 30
    start_time = time.time()
    for i in range(0,n,batch_size):
        if (i/batch_size)%1==0:
            print 'neighbor:',offset+i,'/',n,'median distance=',neighbor_median(lneighbors,k),'time=',int(time.time() - start_time),'s'
        Xb = X[i:i+min(batch_size,n-i)]
        knn = NearestNeighbors(n_neighbors=k, algorithm='brute',metric='cosine').fit(Xb)
        dist,indx = knn.kneighbors(Xtest)
        for j in range(0,Xtest.shape[0]):
            neighbor_select(lneighbors,j,dist[j,:],indx[j,:]+i+offset,neighbor_size)
        del Xb,knn
    return lneighbors

neighbor_size = 1000

# parallel neighboring distance computation :
# FIXME : very strange that this did not improve speed !!!
#step = 1000000
#lneighbors = Parallel(n_jobs=2)(delayed(neighbor_distance)(Xtrain[i:min(train_count,i+1000000)],Xtest,i,neighbor_size) for i in range(0,train_count,1000000))

neighbors = neighbor_distance(Xtrain,Xtest,offset=0,neighbor_size=1000)

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
    valid_ids += random.sample(IDneighbor[i,5:20],1)

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

#add for each selected training sample the min distance to test set
ID2D = {}
for i in range(IDneighbor.shape[0]):
    for j in range(IDneighbor.shape[1]):
        ID2D[IDneighbor[i,j]] = min(ID2D.get(IDneighbor[i,j],1),Dneighbor[i,j])

dftrain['D'] = map(lambda i:ID2D[i],dftrain.Identifiant_Produit)

dfsample = training_sample_perfect(dftrain,N=647,class_ratio=class_ratio)

dfsample.to_csv(ddir+'training_perfect.csv',sep=';',index=False,header=False)

####################
# sanity check :
####################

sid = dfsample.Identifiant_Produit.values
vid = dfvalid.Identifiant_Produit.values

# no duplicate in validation set
assert len(vid) == len(set(vid))

# no intersection between training and validation set
assert not set(sid).intersection(set(vid))

# ensuring Categorie are not too deplated
df = dfsample[['Identifiant_Produit','Categorie3']].drop_duplicates()
cc = df.groupby('Categorie3').count().values
print 'mean   =',np.mean(cc)
print 'median =',np.percentile(cc,50)
print '1/10   =',np.percentile(cc,10)
print '9/10   =',np.percentile(cc,90)

