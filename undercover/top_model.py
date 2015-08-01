#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

from utils import wdir,ddir,header,normalize_file,add_txt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from os.path import basename

from utils import itocat1,itocat2,itocat3
from utils import cat1toi,cat2toi,cat3toi
from utils import cat3tocat2,cat3tocat1,cat2tocat1
from utils import cat1count,cat2count,cat3count
from utils import training_sample
from os.path import isfile

from joblib import Parallel, delayed
import time

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


def training_stage1(dftrain,dfvalid):
    fname = ddir + 'joblib/stage1'
    print '-'*50
    print 'training',basename(fname)
    df = dftrain
    dfv = dfvalid
    vec,X = vectorizer(df.txt)
    Y = df['Categorie1'].values
    cla = LogisticRegression(C=5)
    cla.fit(X,Y)
    labels = np.unique(df.Categorie1)
    Xv = vec.transform(dfv.txt)
    Yv = dfv['Categorie1'].values
    sct = cla.score(X[:10000],Y[:10000])
    scv = cla.score(Xv,Yv)
    joblib.dump((labels,vec,cla),fname)
    del X,Y,Xv,Yv,vec,cla
    return sct,scv

def training_stage3(dftrain,dfvalid,cat,i):
    fname = ddir + 'joblib/stage3_'+str(cat)
    df = dftrain[dftrain.Categorie1 == cat].reset_index(drop=True)
    dfv = dfvalid[dfvalid.Categorie1 == cat].reset_index(drop=True)
    labels = np.unique(df.Categorie3)
    if len(labels)==1:
        joblib.dump((labels,None,None),fname)
        scv = -1
        sct = -1
        print 'training',cat,'\t\t(',i,') : N=',len(df),'K=',len(labels)
        print 'training',cat,'\t\t(',i,') : training=',sct,'validation=',scv
        return (sct,scv)
    vec,X = vectorizer(df.txt)
    Y = df['Categorie3'].values
    cla = LogisticRegression(C=5)
    cla.fit(X,Y)
    labels = np.unique(df.Categorie3)
    sct = cla.score(X[:min(10000,len(df))],Y[:min(10000,len(df))])
    if len(dfv)==0:
        scv = -1
    else:
        Xv = vec.transform(dfv.txt)
        Yv = dfv['Categorie3'].values
        scv = cla.score(Xv,Yv)
    print 'training',cat,'\t\t(',i,') : N=',len(df),'K=',len(labels)
    print 'training',cat,'\t\t(',i,') : training=',sct,'validation=',scv
    joblib.dump((labels,vec,cla),fname)
    del vec,cla
    return (sct,scv)

#####################
# create sample set 
# from training set
#####################

# NOTE : USE analyse_test.py and perfect_sampling.py to get perfect training & validation set
# NOTE : USE analyse_test.py and perfect_sampling.py to get perfect training & validation set
# NOTE : USE analyse_test.py and perfect_sampling.py to get perfect training & validation set
# NOTE : USE analyse_test.py and perfect_sampling.py to get perfect training & validation set

#######################
# training
# stage1 : Categorie1 
# stage3 : Categorie3|Categorie1
#######################

dftrain = pd.read_csv(ddir+'training_perfect.csv',sep=';',names = header()+['D',],).fillna('')
dfvalid = pd.read_csv(ddir+'validation_perfect.csv',sep=';',names = header()).fillna('')
dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')

add_txt(dftrain)
add_txt(dfvalid)
add_txt(dftest)

dftrain = dftrain[['Categorie3','Categorie1','txt']]
dfvalid = dfvalid[['Categorie3','Categorie1','txt']]
dftest = dftest[['Identifiant_Produit','txt']]

# training stage1

dt = -time.time()
sct,scv = training_stage1(dftrain,dfvalid)
dt += time.time()

print '##################################'
print '# stage1 elapsed time :',dt
print '# stage1 training score :',sct
print '# stage1 validation score :',scv
print '##################################'


# training parralel stage3
cat1 = np.unique(dftrain.Categorie1)
#training_stage3(ctx,cat1[0])
dfts = []
dfvs = []
for cat in cat1:
    dfts.append(dftrain[dftrain.Categorie1 == cat].reset_index(drop=True))
    dfvs.append(dfvalid[dfvalid.Categorie1 == cat].reset_index(drop=True))


dt = -time.time()
scs = Parallel(n_jobs=3)(delayed(training_stage3)(dft,dfv,cat,i) for i,(dft,dfv,cat) in enumerate(zip(dfts,dfvs,cat1)))
dt += time.time()

sct = np.median([s for s in zip(*scs)[0] if s>=0])
scv = np.median([s for s in zip(*scs)[1] if s>=0])

print '##################################'
print '# stage3 elapsed time :',dt
print '# stage3 training score :',sct
print '# stage3 validation score :',scv
print '##################################'

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

dfvalid = pd.read_csv(ddir+'validation_perfect.csv',sep=';',names = header()).fillna('').reset_index()
dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')

add_txt(dfvalid)
add_txt(dftest)

#######################
# stage 1 log proba filling
#######################
stage1_log_proba_valid = np.full(shape=(len(dfvalid),cat1count),fill_value = -666.,dtype = float)
stage1_log_proba_test = np.full(shape=(len(dftest),cat1count),fill_value = -666.,dtype = float)

fname = ddir + 'joblib/stage1'
(labels,vec,cla) = joblib.load(fname)
(classes,lpv) = log_proba(dfvalid,vec,cla)
(classes,lpt) = log_proba(dftest,vec,cla)
for i,k in enumerate(classes):
    j = cat1toi[k]
    stage1_log_proba_valid[:,j] = lpv[:,i]
    stage1_log_proba_test[:,j] = lpt[:,i]

del labels,vec,cla


#######################
# stage 3 log proba filling
#######################
stage3_log_proba_valid = np.full(shape=(len(dfvalid),cat3count),fill_value = -666.,dtype = float)
stage3_log_proba_test = np.full(shape=(len(dftest),cat3count),fill_value = -666.,dtype = float)

for ii,cat in enumerate(itocat1):
    fname = ddir + 'joblib/stage3_'+str(cat)
    print '-'*50
    print 'predicting',basename(fname),':',ii,'/',len(itocat1)
    if not isfile(fname): 
        continue
    (labels,vec,cla) = joblib.load(fname)
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
    del labels,vec,cla

print '>>> dump stage1 & stage2 log_proba'
joblib.dump((stage1_log_proba_valid,stage3_log_proba_valid),ddir+'/joblib/log_proba_valid')
joblib.dump((stage1_log_proba_test,stage3_log_proba_test),ddir+'/joblib/log_proba_test')
print '<<< dump stage1 & stage2 log_proba'


##################
# (stage1_log_proba_valid,stage3_log_proba_valid) = joblib.load(ddir+'/joblib/log_proba_valid')
# (stage1_log_proba_test,stage3_log_proba_test) = joblib.load(ddir+'/joblib/log_proba_test')
##################

## FIXME
## FIXME
## FIXME
## >>>>>
#
## greedy approach:
#(stage1_log_proba_valid,stage3_log_proba_valid) = joblib.load(fname)
#
##predict_cat1_valid = [itocat1[i] for i in np.argmax(stage1_log_proba_valid,axis=1)]
#
#for i,cat1 in enumerate(predict_cat1_valid):
#    if i%1000==0:
#        print 1.*i/len(predict_cat1_valid)
#    for j in [k for k,cat3 in enumerate(itocat3) if cat3tocat1[cat3] != cat1]:
#        stage3_log_proba_valid[i,j] = -666
#
#predict_cat3_valid = [itocat3[i] for i in np.argmax(stage3_log_proba_valid,axis=1)]
#
#score_cat1 = sum(dfvalid.Categorie1 == predict_cat1_valid)*1.0/len(dfvalid)
#score_cat3 = sum(dfvalid.Categorie3 == predict_cat3_valid)*1.0/len(dfvalid)
#print 'dfvalid scores =',score_cat1,score_cat3
#
## <<<<<
## FIXME
## FIXME
## FIXME

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
# candidate top model    #
##########################

##################################
# NOTE : perfect training & validation on top's 456 NN
# NOTE : C=5,C=5,max_features=234567
##################################
# stage1 elapsed time : 3598.16943312
# stage1 training score : 0.9804
# stage1 validation score : 0.874329215026
# stage3 elapsed time : 3542.61673594
# stage3 training score : 0.9893
# stage3 validation score : 0.849927849928
# validation score : 0.874329215026 0.687476600524
# (resultat44.csv) test score : 66,39161% (NOTE TOP TOP TOP)
##################################

##################################
# NOTE : perfect training & validation on top's 456 NN
# NOTE : C=3,C=3,max_features=234567
##################################
# stage1 elapsed time : 3175.98400712
# stage1 training score : 0.976
# stage1 validation score : 0.871614875827
# stage3 elapsed time : 3369.5460999
# stage3 training score : 0.984
# stage3 validation score : 0.843361986628
# validation score : 0.871614875827 0.680643953575
# (resultat45.csv) test score : 66,32858%
##################################

##################################
# NOTE : perfect training & validation on top's 456 NN
# NOTE : C=11,C=11,max_features=234567
##################################
# stage1 elapsed time : 4176.42304993
# stage1 training score : 0.9864
# stage1 validation score : 0.875920379384
# stage3 elapsed time : 3805.47599411
# stage3 training score : 0.9959
# stage3 validation score : 0.854256854257
# validation score : 0.875920379384 0.694122051666
# (resultat46.csv) test score : 66,23116%.
##################################

##################################
# NOTE : perfect training & validation on top's 456 NN
# NOTE : C=8,C=8,max_features=234567
##################################
# stage1 elapsed time : 3918.26019287
# stage1 training score : 0.9846
# stage1 validation score : 0.875857980781
# stage3 elapsed time : 3714.93741202
# stage3 training score : 0.99375
# stage3 validation score : 0.853535353535
# validation score : 0.875857980781 0.692250093598
# (resultat47.csv) test score : 66,32285%
##################################

##################################
# NOTE : perfect training & validation on top's 456 NN
# NOTE : C=20,C=5,max_features=234567
##################################
# stage1 elapsed time : 4794.11950397
# stage1 training score : 0.9898
# stage1 validation score : 0.876731561213
# stage3 elapsed time : 3549.80708003
# stage3 training score : 0.9893
# stage3 validation score : 0.849927849928
# validation score : 0.876731561213 0.691189317359
# (resultat48.csv) test score : 66,10509%
##################################

