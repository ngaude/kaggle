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
from utils import header,add_txt
from utils import itocat1,itocat2,itocat3
from utils import cat1toi,cat2toi,cat3toi
from utils import cat3tocat2,cat3tocat1,cat2tocat1
from utils import cat1count,cat2count,cat3count
from utils import MarisaTfidfVectorizer
from utils import adasyn_sample
from sklearn.preprocessing import normalize

def tfidf_vectorizer(txt):
    vec = MarisaTfidfVectorizer(
        min_df = 2,
        max_features = 1000000,
        stop_words = None,
        smooth_idf=True,
        norm='l2',
        sublinear_tf=True,
        use_idf=True,
        ngram_range=(1,2))
    vec.fit(txt)
    return vec

def rf_vectorizer(df):
    assert 'D' in df.columns
    assert 'proba_lr' in df.columns
    assert 'num_word_test' in df.columns
    assert 'num_word_nn' in df.columns
    X = np.zeros(shape=(len(df),4))
    X[:,0] = df.D
    X[:,1] = df.proba_lr
    X[:,2] = df.num_word_test
    X[:,3] = df.num_word_nn
    X = normalize(X,axis=0)
    Y = np.array(df.Y.values)
    return (X,Y)

def training_sample_adasyn(df,n = 200,mincount=7):
    (X,Y) = rf_vectorizer(df)
    Xt = []
    Yt = []
    for i,cat in enumerate(np.unique(df.Y)):
        print 'adasyn :',i
        Xt.append(adasyn_sample(X,Y,cat,K=5,n=n))
        Yt.append([cat,]*Xt[-1].shape[0])
    Xt = np.vstack(Xt) 
    Yt = np.concatenate(Yt)
    shuffle = np.random.permutation(len(Yt))
    Xt = Xt[shuffle,:]
    Yt = Yt[shuffle]
    return Xt,Yt

def training_sample_random(df,n = 200,mincount=7):
    n = int(n)
    cl = df.Y
    cc = cl.groupby(cl)
    s = (cc.count() >= mincount)
    labelmaj = s[s].index
    print 'sampling =',n,'samples for any of',len(labelmaj),'classes'
    dfs = []
    for i,cat in enumerate(labelmaj):
        if i%100==0:
            print 'sampling',i,'/',len(labelmaj),':'
        dfcat = df[df.Y == cat]
        sample_count = n
        if len(dfcat)>=sample_count:
            # undersample sample_count samples : take the closest first
            rows = random.sample(dfcat.index, sample_count)
            dfs.append(dfcat.ix[rows])
        else:
            # sample all samples + oversample the remaining
            dfs.append(dfcat)
            dfcat = dfcat.iloc[np.random.randint(0, len(dfcat), size=sample_count-len(dfcat))]
            dfs.append(dfcat)
    dfsample = pd.concat(dfs)
    dfsample = dfsample.reset_index(drop=True)
    dfsample = dfsample.reindex(np.random.permutation(dfsample.index),copy=False)
    return rf_vectorizer(dfsample)

ddir = '/home/ngaude/workspace/data/cdiscount.auto/'

#############################################
# get results from a previously trained 
# logistic regression model
#############################################

valid = pd.read_csv(ddir+'validation_sample.csv.100',names=header(),sep=';').fillna('')
(stage1_log_proba_valid,stage3_log_proba_valid) = joblib.load(ddir+'/joblib/log_proba_valid.100')

def bayes_prediction(stage1_log_proba,stage3_log_proba):
    for i in range(stage3_log_proba.shape[1]):
        cat3 = itocat3[i]
        cat1 = cat3tocat1[cat3]
        j = cat1toi[cat1]
        stage3_log_proba[:,i] += stage1_log_proba[:,j]

bayes_prediction(stage1_log_proba_valid,stage3_log_proba_valid)

predict_cat3_valid = [itocat3[i] for i in np.argmax(stage3_log_proba_valid,axis=1)]
proba_cat3_valid =  np.exp(np.max(stage3_log_proba_valid,axis=1))

valid['Categorie3_lr'] = predict_cat3_valid
valid['proba_lr'] = proba_cat3_valid
add_txt(valid)

#############################################
# get results from a previously trained 
# logistic regression model
#############################################

# head = pd.read_csv(ddir+'training_head.csv',names=header(),sep=';').fillna('')
# head = head[head.Produit_Cdiscount == 1]
# add_txt(head)
# head.to_csv(ddir+'nn_train.csv',sep=';',index=False)

train = pd.read_csv(ddir+'nn_train.csv',sep=';')

#############################################
# vectorize the full text                   #
#############################################

vec = tfidf_vectorizer(pd.concat([valid.txt,train.txt]))

Xtest = vec.transform(valid.txt)
Xtrain = vec.transform(train.txt)

##################
# FIND NEAREST NEIGHBORS
##################

train_count = Xtrain.shape[0]
test_count = Xtest.shape[0]

nn = [(1,0)]*test_count

def nn_median(nn):
    return np.median([ tup[0] if tup else 1 for tup in nn])

batch_size = 10000
start_time = time.time()
for i in range(0,train_count,batch_size):
    if (i/batch_size)%1==0:
        print 'neighbor:',i,'/',train_count,'median distance=',nn_median(nn),'time=',float(time.time() - start_time),'s'
    X = Xtrain[i:i+min(batch_size,train_count-i)]
    knn = NearestNeighbors(n_neighbors=1, algorithm='brute',metric='cosine').fit(X)
    dist,indx = knn.kneighbors(Xtest)
    nn = [ (dist[j,0],indx[j,0]+i) if dist[j,0]<tup[0] else tup for j,tup in enumerate(nn) ]

##########################
# JOIN VALID WITH NEIGHBORS
##########################

valid['D'] = zip(*nn)[0]
valid['nn'] = map(lambda i: train.loc[i].Identifiant_Produit, zip(*nn)[1])

df = valid.merge(train,'left',None,'nn','Identifiant_Produit',suffixes=('_test', '_nn'))

df['num_word_test'] = map(lambda t:len(set(t.split())),df.txt_test)
df['num_word_nn'] = map(lambda t:len(set(t.split())),df.txt_nn)
df.to_csv('valid_nn_training.csv',sep=';',index=False)


##################
##################
##################
##################
##################
##################
##################
##################
# TRAIN RF TO FIND OK NEAREST NEIGHBORS 
# THAT WOULD HAVE NOT BEEN FOUND BY
# LOGIT :
# 332-NN vs 19356-LR ...
##################
##################
##################
##################
##################
##################
##################


df = pd.read_csv('valid_nn_training.csv',sep=';')

def nn_label(r):
    lr_ok = (r.Categorie3_lr == r.Categorie3_test)
    nn_ok = (r.Categorie3_nn == r.Categorie3_test)
    if lr_ok and nn_ok:
        return 3
    if lr_ok:
        return 2
    if nn_ok:
        return 1
    return 0

df['Y'] = df.apply(nn_label,axis=1)


N = int(5*min([len(g) for i,g in df.groupby('Y')]))

#(X,Y) = training_sample_random(df,N)

(X,Y) = training_sample_adasyn(df,N)


# 
# df_none_sub = pd.DataFrame(df.ix[np.random.choice(f_none,N)])
# df_both_sub = pd.DataFrame(df.ix[np.random.choice(f_both,N)])
# df_lr_only_sub = pd.DataFrame(df.ix[np.random.choice(f_lr_only,N)])
# df_nn_only_sub = pd.DataFrame(df.ix[np.random.choice(f_nn_only,N)])
# 
# df_none_sub['Y'] = 0
# df_nn_only_sub['Y'] = 1
# df_lr_only_sub['Y'] = 2
# df_both_sub['Y']=3
# 
# sdf = pd.concat([df_none_sub,df_both_sub,df_lr_only_sub,df_nn_only_sub]).reset_index(drop=True)
# sdf = sdf.reindex(np.random.permutation(sdf.index))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.15)

cla = RandomForestClassifier(n_estimators = 1000,verbose=1)
cla.fit(Xtrain,Ytrain)
Ypred = cla.predict(Xtest)

print 'F1 score',f1_score(Ytest,Ypred,average=None),'(score=',cla.score(Xtest,Ytest),')'

#joblib.dump(cla,'NN_vs_LR_Classifier')

##################
# PREDICT FROM BEST RESULTAT and NN :
# 0 : LR and NN are FALSE
# 1 : NN is OK
# 2 : LR is OK
# 3 : NN & LR are OK
##################

test_normed = pd.read_csv('/home/ngaude/workspace/data/cdiscount.proba/test_normed.csv',sep=';',names=header(True)).fillna('')
add_txt(test_normed)
num_word_test = map(lambda t:len(set(t.split())),test_normed.txt)

test_nn = pd.read_csv('test_nn.csv',sep=';').fillna('')
test_nn['Marque'] = test_nn.Marque_nn
test_nn['Libelle'] = test_nn.Libelle_nn
test_nn['Description'] = test_nn.Description_nn
add_txt(test_nn)
num_word_nn = map(lambda t:len(set(t.split())),test_nn.txt)
test_nn.drop('Marque', axis=1, inplace=True)
test_nn.drop('Libelle', axis=1, inplace=True)
test_nn.drop('Description', axis=1, inplace=True)

best = pd.read_csv('/home/ngaude/workspace/data/cdiscount.proba/proba.auto.merging.60.csv',sep=';')

nn = test_nn.merge(best,'left',None,'Identifiant_Produit_test','Id_Produit')
nn['num_word_test'] = num_word_test;
nn['num_word_nn'] = num_word_nn;
nn['proba_lr'] = nn.Proba_Categorie3
nn['Y'] = None

(X,Y) = rf_vectorizer(nn)

Y = cla.predict(X)

nn.Y = Y

maxD = np.percentile(nn.D,5)
minP = np.percentile(nn.proba_lr,68)

def choose_Categorie3(r):
    if r.D > maxD:
        return r.Id_Categorie
    if r.proba_lr > minP:
        return r.Id_Categorie
    if r.Y == 0:
        return r.Id_Categorie
    if r.Y == 1:
        return r.Categorie3
    if r.Y == 2:
        return r.Id_Categorie
    if r.Y == 3:
        return r.Id_Categorie
    return r.Id_Categorie

nn.Id_Categorie = nn.apply(choose_Categorie3,axis=1)

df = nn[['Id_Produit','Id_Categorie']]

df.to_csv('resultat.blended.csv',sep=';',index=False)




