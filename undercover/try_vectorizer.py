#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils import ddir,header,add_txt,MarisaTfidfVectorizer,adasyn_sample,cat3tocat1
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from os.path import basename
import time
import random
from joblib import Parallel, delayed

def vectorizer(txt):
    vec = MarisaTfidfVectorizer(
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

def get_sample(dftrain,mincount = 9,maxcount = 647):
    cl = dftrain['Categorie3']
    cc = cl.groupby(cl)
    s = (cc.count() > mincount)
    labelmaj = s[s].index
    print len(labelmaj)
    dfs = []
    for i,cat in enumerate(labelmaj):
        if i%100==0:
            print i,'/',len(labelmaj),':'
        df = dftrain[dftrain['Categorie3'] == cat]
        if len(df) > maxcount:
            # undersample the majority samples
            rows = random.sample(df.index, maxcount)
            dfs.append(df.ix[rows])
        else:
            # sample all the minority samples
            dfs.append(df)
    dfsample = pd.concat(dfs)
    dfsample = dfsample.reset_index(drop=True)
    dfsample = dfsample.reindex(np.random.permutation(dfsample.index),copy=False)
    return dfsample

print 'loading...'
dftrain = pd.read_csv(ddir+'training_shuffled_normed.csv',sep=';',names = header()).fillna('')
print 'txting...'
add_txt(dftrain)

dfsample = get_sample(dftrain)
dfsample = dfsample[['Identifiant_Produit','Categorie3','Categorie1','txt']]

dfsample.to_csv(ddir+'training_sup9.csv',sep=';',index=False,header=False)

Y = dfsample.Categorie3.values
ID = dfsample.Identifiant_Produit.values
print 'vectorizing...'
vec,X = vectorizer(dfsample.txt)
print 'dumping...'
joblib.dump((vec,ID,X,Y),ddir+'joblib/vecIDXY')

# use adasyn to get synthetic balanced dataset

Xt = []
Yt = []
for i,cat in enumerate(np.unique(Y)):
    print 'adasyn :',i
    Xt.append(adasyn_sample(X,Y,cat,K=5,n=200))
    Yt.append([cat,]*Xt[-1].shape[0])

Xt = sparse.vstack(Xt) 
assert Xt.shape[0] == len(Yt)
rows = random.sample(Xt,Xt.shape[0])
Xt = Xt[rows]
joblib.dump((vec,Xs,Ys),ddir+'joblib/vecXtYt')

#################################################
# TRAINING START HERE
#################################################


def training_stage1(X,Y,Xv,Yv):
    fname = ddir + 'joblib/stage1'
    print '-'*50
    print 'training',basename(fname)
    cla = LogisticRegression(C=5)
    cla.fit(X,Y)
    labels = np.unique(Y)
    sct = cla.score(X[:10000],Y[:10000])
    scv = cla.score(Xv,Yv)
    joblib.dump((labels,cla),fname)
    del X,Y,Xv,Yv,cla
    return sct,scv

def training_stage3(X,Y,Xv,Yv,cat,i):
    fname = ddir + 'joblib/stage3_'+str(cat)
    labels = np.unique(Y)
    if len(labels)==1:
        print fname,'predict 100% ',labels[0]
        joblib.dump((labels,None,None),fname)
        scv = -1
        sct = -1
        return (sct,scv)
    cla = LogisticRegression(C=5)
    cla.fit(X,Y)
    sct = cla.score(X[:min(10000,len(df))],Y[:min(10000,len(df))])
    if len(dfv)==0:
        scv = -1
    else:
        scv = cla.score(Xv,Yv)
    print 'Stage 3.'+str(i)+':',cat,'score',sct,scv
    joblib.dump((labels,cla),fname)
    del cla
    return (sct,scv)

(vec,ID,X,Y) = joblib.load(ddir+'joblib/vecIDXY')
dfvalid = pd.read_csv(ddir+'validation_normed.csv',sep=';',names = header()).fillna('')
add_txt(dfvalid)
Xv = vec.transform(dfvalid.txt)
Yv = dfvalid.Categorie3
dt = -time.time()
sct,scv = training_stage1(X,Y,Xv,Yv)
dt += time.time()

print '**********************************'
print 'stage1 elapsed time :',dt
print 'stage1 training score :',sct
print 'stage1 validation score :',scv
print '**********************************'

# training parallel stage3

Z = map(lambda c:cat3tocat1[c], Y)
Zv = map(lambda c:cat3tocat1[c], Yv)
cat1 = np.unique(Z)
XYs = []
XYvs = []

for cat in cat1:
    indices = np.nonzero(Z==cat)
    XYs.append((X[indices],Y[indices]))
    indices = np.nonzero(Zv==cat)
    XYvs.append((Xv[indices],Yv[indices]))

dt = -time.time()
scs = Parallel(n_jobs=3)(delayed(training_stage3)(XY[0],XY[1],XYv[0],XYv[1],cat,i) for i,(XY,XYv,cat) in enumerate(zip(XYs,XYvs,cat1)))
dt += time.time()

sct = np.median([s for s in zip(*scs)[0] if s>=0])
scv = np.median([s for s in zip(*scs)[1] if s>=0])

print '**********************************'
print 'stage3 elapsed time :',dt
print 'stage3 training score :',sct
print 'stage3 validation score :',scv
print '**********************************'

#######################
# predicting
#######################

# predict : 
# P(Categorie3) = P(Categorie1) *  P(Categorie3|Categorie1)

def log_proba(df,vec,cla):
    assert 'txt' in df.columns
    X = vec.transform(df.txt)
    lp = cla.predict_log_proba(X)
    return (cla.classes_,lp)

dfvalid = pd.read_csv(ddir+'validation_normed.csv',sep=';',names = header()).fillna('')
add_txt(dfvalid)
Xv = vec.transform(dfvalid.txt)
Yv = dfvalid.Categorie3

dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header()).fillna('')
add_txt(dftest)
Xt = vec.transform(dftest.txt)
Yt = dftest.Categorie3

#######################
# stage 1 log proba filling
#######################
stage1_log_proba_valid = np.full(shape=(Xv.shape[0],cat1count),fill_value = -666.,dtype = float)
stage1_log_proba_test  = np.full(shape=(Xf.shape[0],cat1count),fill_value = -666.,dtype = float)

fname = ddir + 'joblib/stage1'
(labels,cla) = joblib.load(fname)
(classes,lpv) = log_proba(dfvalid,vec,cla)
(classes,lpt) = log_proba(dftest,vec,cla)
for i,k in enumerate(classes):
    j = cat1toi[k]
    stage1_log_proba_valid[:,j] = lpv[:,i]
    stage1_log_proba_test[:,j] = lpt[:,i]

del labels,cla


#######################
# stage 3 log proba filling
#######################
stage3_log_proba_valid = np.full(shape=(Xv.shape[0],cat3count),fill_value = -666.,dtype = float)
stage3_log_proba_test  = np.full(shape=(Xf.shape[0],cat3count),fill_value = -666.,dtype = float)

for ii,cat in enumerate(itocat1):
    fname = ddir + 'joblib/stage3_'+str(cat)
    print '-'*50
    print 'predicting',basename(fname),':',ii,'/',len(itocat1)
    if not isfile(fname): 
        continue
    (labels,cla) = joblib.load(fname)
    if len(labels)==1:
        k = labels[0]
        j = cat3toi[k]
        stage3_log_proba_valid[:,j] = 0
        stage3_log_proba_test[:,j] = 0
        continue
    (classes,lpv) = log_proba(dfvalid,vec,cla)
    (classes,lpt) = log_proba(dftest,vec,cla)
    for i,k in enumerate(classes):
        j = cat3toi[k]
        stage3_log_proba_valid[:,j] = lpv[:,i]
        stage3_log_proba_test[:,j] = lpt[:,i]
    del labels,cla

print '>>> dump stage1 & stage2 log_proba'
joblib.dump((stage1_log_proba_valid,stage3_log_proba_valid),ddir+'/joblib/log_proba_valid')
joblib.dump((stage1_log_proba_test,stage3_log_proba_test),ddir+'/joblib/log_proba_test')
print '<<< dump stage1 & stage2 log_proba'


##################
# (stage1_log_proba_valid,stage3_log_proba_valid) = joblib.load(ddir+'/joblib/log_proba_valid')
# (stage1_log_proba_test,stage3_log_proba_test) = joblib.load(ddir+'/joblib/log_proba_test')
##################

##################
# bayes rulez ....
##################

assert stage3_log_proba_valid.shape[1] == stage3_log_proba_test.shape[1]

for i in range(stage3_log_proba_valid.shape[1]):
    cat3 = itocat3[i]
    cat1 = cat3tocat1[cat3]
    j = cat1toi[cat1]
    stage3_log_proba_valid[:,i] += stage1_log_proba_valid[:,j]
    stage3_log_proba_test[:,i] += stage1_log_proba_test[:,j]

predict_cat1_valid = [itocat1[i] for i in np.argmax(stage1_log_proba_valid,axis=1)]
predict_cat3_valid = [itocat3[i] for i in np.argmax(stage3_log_proba_valid,axis=1)]
predict_cat1_test = [itocat1[i] for i in np.argmax(stage1_log_proba_test,axis=1)]
predict_cat3_test = [itocat3[i] for i in np.argmax(stage3_log_proba_test,axis=1)]

def submit(df,Y):
    submit_file = ddir+'resultat.csv'
    df['Id_Produit']=df['Identifiant_Produit']
    df['Id_Categorie'] = Y
    df= df[['Id_Produit','Id_Categorie']]
    df.to_csv(submit_file,sep=';',index=False)

score_cat1 = sum(dfvalid.Categorie1 == predict_cat1_valid)*1.0/len(dfvalid)
score_cat3 = sum(dfvalid.Categorie3 == predict_cat3_valid)*1.0/len(dfvalid)
print 'validation score :',score_cat1,score_cat3
submit(dftest,predict_cat3_test)

##########################
# reference model score  #
##########################
# NOTE
# NOTE
# NOTE
# NOTE
#################################################
# stage1 elapsed time : 966.20471406
# stage1 training score : 0.9633
# stage1 validation score : 0.874375015096
# stage3 elapsed time : ~1500
# stage3 training score : 0.984027777778
# stage3 validation score : 0.863612147043
# validation score : 0.874375015096 0.68585299872
# (result30.csv) test score : 63,99060%
#################################################
#
#################################################
# stage1 elapsed time : 448.81675601
# stage1 training score : 0.9569
# stage1 validation score : 0.874882249221
# stage3 elapsed time : 765.464223146
# stage3 training score : 0.9839
# stage3 validation score : 0.857221006565
# validation score : 0.874882249221 0.684452066375
# (result31.csv) test score : 64,01352%
#################################################

