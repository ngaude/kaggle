import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

#### where data come from ...
#### (train.csv) => (train.Rda) =>  (train.featured.csv) with xgb.R
#### (train.featured.csv) => (joblib/XY_train) with rf.py
# 
# train = pd.read_csv('./data/train.featured.csv')
# X = train.as_matrix()
# del(train)
# n = X.shape[0]
# Y = X[:,0]
# ID = X[:,1]
# X = X[:,2:]
# joblib.dump((ID,X,Y),'./data/joblib/XY_train')
# 
# test = pd.read_csv('./data/test.featured.csv')
# X = test.as_matrix()
# del(test)
# n = X.shape[0]
# ID = X[:,1]
# X = X[:,2:]
# joblib.dump((ID,X),'./data/joblib/X_test')
# 
############################################
# RF
############################################

ID_test,X_test = joblib.load('./data/joblib/X_test')
ID_train,X_train,Y_train = joblib.load('./data/joblib/XY_train')

# optional CV split...
X, X_cv, Y, Y_cv = train_test_split(X,Y,random_state=42,test_size=0.05)

# default use it all...
X = X_train
Y = Y_train


clf = RandomForestClassifier(verbose=True,n_estimators=100,random_state=42,class_weight='auto',n_jobs=4)
clf.fit(X,Y)
Y_pred = clf.predict_proba(X_cv)[:,1]
score = roc_auc_score(Y_cv,Y_pred)
print 'forest score:',score

Y_test = clf.predict_proba(X_test)[:,1]
df = pd.DataFrame({'ID': map(int,ID_test), 'target': Y_test})
df.to_csv('./data/rf_nikko_1.csv',index=False)


# forest 1000 trees score: cv=0.77235702726

# def balanced_set(X,Y,n=15000):
#     X_pos = X[Y == 1]
#     X_neg = X[Y == 0]
#     id_pos = np.random.choice(len(X_pos),n,replace=False)
#     id_neg = np.random.choice(len(X_neg),n,replace=False)
#     shuf = np.random.choice(2*n,2*n,replace=False)
#     X_balanced = np.vstack((X_pos[id_pos],X_neg[id_neg]))
#     Y_balanced = np.array([1]*n+[0]*n)
#     X_balanced = X_balanced[shuf,:]
#     Y_balanced = Y_balanced[shuf]
#     return (X_balanced,Y_balanced)
# 
# n_forest = 10
# Y_preds = np.zeros(shape=(len(Y_cv),n_forest))
# 
# for i in range(n_forest):
#     n_samples = int(sum(Y)*0.5)
#     X_balanced,Y_balanced = balanced_set(X,Y,n_samples)
#     clf = RandomForestClassifier(verbose=True,n_estimators=100,random_state=42)
#     clf.fit(X_balanced,Y_balanced)
#     Y_pred = clf.predict_proba(X_cv)[:,1]
#     score = roc_auc_score(Y_cv,Y_pred)
#     print 'forest',i,'score:',score
#     Y_preds[:,i] = Y_pred
# 
# Y_pred = np.median(Y_preds,axis=1)
# 
# score = roc_auc_score(Y_cv,Y_pred)
# 
# print n_forest,'x forest score:',score
## 10 x forest 100 x trees score (~75.4) : 0.762517459312




