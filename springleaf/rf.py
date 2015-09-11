import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

train = pd.read_csv('./data/train.featured.csv')

X = train.as_matrix()
del(train)
n = X.shape[0]
Y = X[:,0]
X = X[:,2:]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=42)

clf = RandomForestClassifier(verbose=True,n_estimators=1000,random_state=42,class_weight='auto',n_jobs=4)
clf.fit(X_train,Y_train)
Y_pred = clf.predict_proba(X_test)[:,1]
score = roc_auc_score(Y_test,Y_pred)
print 'forest score:',score

# forest 1000 trees score: cv=0.77235702726


def balanced_set(X_train,Y_train,n=15000):
    X_pos = X_train[Y_train == 1]
    X_neg = X_train[Y_train == 0]
    id_pos = np.random.choice(len(X_pos),n,replace=False)
    id_neg = np.random.choice(len(X_neg),n,replace=False)
    shuf = np.random.choice(2*n,2*n,replace=False)
    X_balanced = np.vstack((X_pos[id_pos],X_neg[id_neg]))
    Y_balanced = np.array([1]*n+[0]*n)
    X_balanced = X_balanced[shuf,:]
    Y_balanced = Y_balanced[shuf]
    return (X_balanced,Y_balanced)

n_forest = 10
Y_preds = np.zeros(shape=(len(Y_test),n_forest))

for i in range(n_forest):
    n_samples = int(sum(Y_train)*0.5)
    X_balanced,Y_balanced = balanced_set(X_train,Y_train,n_samples)
    clf = RandomForestClassifier(verbose=True,n_estimators=100,random_state=42)
    clf.fit(X_balanced,Y_balanced)
    Y_pred = clf.predict_proba(X_test)[:,1]
    score = roc_auc_score(Y_test,Y_pred)
    print 'forest',i,'score:',score
    Y_preds[:,i] = Y_pred

Y_pred = np.median(Y_preds,axis=1)

score = roc_auc_score(Y_test,Y_pred)

print n_forest,'x forest score:',score
## 10 x forest 100 x trees score: 0.753502231719




