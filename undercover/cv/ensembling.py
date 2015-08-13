import numpy as np
import pandas as pd
from utils import cat1count,itocat1,cat1toi,cat3tocat1
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
ddir = '/home/ngaude/workspace/data/cdiscount/'

lr = pd.read_csv(ddir+'proba.auto.valid.cv.csv',sep=';').drop_duplicates()
nn = pd.read_csv(ddir+'confidence.nn.valid.cv.csv',sep=';').drop_duplicates()
valid = pd.read_csv(ddir+'valid_cv.csv',sep=';').fillna('').drop_duplicates()

valid = valid[['Identifiant_Produit','Categorie3']]

df = valid.merge(lr,'left',None,'Identifiant_Produit','Id_Produit')

df = df.merge(nn,'left',None,'Identifiant_Produit','Id_Produit',suffixes=['_lr','_nn'])

def choose_categorie(r):
    if r.Ypred==1:
        return r.Id_Categorie_nn
    return r.Id_Categorie_lr

def meta_vectorizer(df):
    n = len(df)
    X = np.zeros(shape=(n,4+cat1count))
    X[:,0] = (df.Proba_Categorie1)/df.Proba_Categorie1.mean()
    X[:,1] = (df.Proba_Categorie3)/df.Proba_Categorie3.mean()
    X[:,2] = (df.confidence)/df.confidence.mean()
    X[:,3] = df.D/df.D.mean()
    for i,cat3 in enumerate(df.Id_Categorie_lr):
        j = 4 + cat1toi[cat3tocat1[cat3]]
        X[i,j] += 1.
    for i,cat3 in enumerate(df.Id_Categorie_nn):
        j = 4 + cat1toi[cat3tocat1[cat3]]
        X[i,j] += 1.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X


"""
Identifiant_Produit
Categorie3
Id_Produit_lr
Id_Categorie_lr
Proba_Categorie1
Proba_Categorie3
Id_Produit_nn
Id_Categorie_nn
confidence
D
"""

X = meta_vectorizer(df)

nn_only = (df.Id_Categorie_nn != df.Id_Categorie_lr) & (df.Id_Categorie_nn == df.Categorie3)
print 'max possible ensembling improvment is:', np.mean(nn_only)

# Y is the target to obtain !!!
Y = nn_only

#####################
# CV the CV
Xa = X[:30000,:]
Xb = X[30000:,:]
Ya = Y[:30000]
Yb = Y[30000:]
cvcv = pd.DataFrame(df.iloc[30000:])

cla = RandomForestClassifier(n_estimators=100,verbose=1)
cla.fit(Xa,Ya)
Ypred = cla.predict(Xb)
cvcv['Ypred'] = Ypred
cvcv['Id_Categorie_pred'] = cvcv.apply(choose_categorie,axis=1)

print 'CVCV NN score',np.mean(cvcv.Id_Categorie_nn == cvcv.Categorie3)
print 'CVCV LR score',np.mean(cvcv.Id_Categorie_lr == cvcv.Categorie3)
print 'CVCV PRED score',np.mean(cvcv.Id_Categorie_pred == cvcv.Categorie3)




####################

cla = RandomForestClassifier(n_estimators=100,verbose=1)
cla.fit(X,Y)
Ypred = cla.predict(X)

df['Ypred']=Ypred


df['Id_Categorie_pred'] = df.apply(choose_categorie,axis=1)


print 'NN score',np.mean(df.Id_Categorie_nn == df.Categorie3)
print 'LR score',np.mean(df.Id_Categorie_lr == df.Categorie3)
print 'PRED score',np.mean(df.Id_Categorie_pred == df.Categorie3)

#################################
# TEST our meta-ensemble on test
#################################

lr = pd.read_csv(ddir+'proba.auto.merging.70.csv',sep=';')
nn = pd.read_csv(ddir+'confidence.nn.5NN.csv',sep=';')
test = pd.read_csv(ddir+'test.csv',sep=';').fillna('')

df = test[['Identifiant_Produit']]

df = df.merge(lr,'left',None,'Identifiant_Produit','Id_Produit')
df = df.merge(nn,'left',None,'Identifiant_Produit','Id_Produit',suffixes=['_lr','_nn'])


X = meta_vectorizer(df)

Ypred = cla.predict(X)

df['Ypred']=Ypred

df['Id_Categorie_pred'] = df.apply(choose_categorie,axis=1)

print 'from NN only',np.mean((df.Id_Categorie_pred == df.Id_Categorie_nn) & (df.Id_Categorie_pred != df.Id_Categorie_lr))
print 'from LR only',np.mean((df.Id_Categorie_pred != df.Id_Categorie_nn) & (df.Id_Categorie_pred == df.Id_Categorie_lr))
print 'from BOTH',np.mean((df.Id_Categorie_pred == df.Id_Categorie_nn) & (df.Id_Categorie_pred == df.Id_Categorie_lr))


df['Id_Produit'] = df.Identifiant_Produit
df['Id_Categorie'] = map(int,df.Id_Categorie_pred)
df = df[['Id_Produit','Id_Categorie']]
df.to_csv(ddir+'resultat.NN.LR.ensembling.csv',sep=';',index=False)


