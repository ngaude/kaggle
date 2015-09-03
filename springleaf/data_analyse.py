import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


ddir = '/home/ngaude/workspace/github/kaggle/springleaf/data/'

df = pd.read_csv(ddir+'train.csv',sep=',')
df = df.fillna(999999999)

uid = df.ID.values
Y = df.target
df = df.drop(['ID','target'],axis=1)

columns_dropped = [c for c in df.columns if df[c].nunique()<2]
columns_int = [c for c in df.columns if df[c].dtype is np.dtype('int')]
columns_float = [c for c in df.columns if df[c].dtype is np.dtype('float')]
columns_string = [c for c in df.columns if df[c].dtype is np.dtype('O')]

print 'columns_dropped =',len(columns_dropped)
print 'columns_int =',len(columns_int)
print 'columns_float =',len(columns_float)
print 'columns_string =',len(columns_string)

# plot the log-count per percentile of distinct numerical values]
a = [df[c].nunique() for c in columns_int+columns_float]
#plt.plot(np.log(np.array([np.percentile(a,i) for i in range(100)])))
#plt.show()
# we decide to cut the numeric categorical vs continuous threshold at 68 percentile.
continuous_count_threshold = int(np.percentile(a,68))
# e.g. 101

assert continuous_count_threshold == 101

columns_numeric = columns_int+columns_float

columns_category = [c for c in columns_numeric if df[c].nunique() <= continuous_count_threshold]

columns_continuous = [c for c in columns_numeric if df[c].nunique() > continuous_count_threshold]

print 'columns_category =',len(columns_category)
print 'columns_continuous =',len(columns_continuous)

# one hot encoding of columns_category
enc = {}
for c in columns_category:
    a = df[c].unique()
    d = {k:v for k,v in zip(a,range(len(a)))}
    enc[c] = d
X_category = df.as_matrix(columns_category)

# linear or log discretization + standardization + variance normalization...
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
X_continuous = df.as_matrix(columns_continuous)
missing_values = [999999994,999999995,999999996,999999997,999999998,999999999,-99999,999999,9999,9998,9994,9995,9996,9997,9998,999,998,997,996,995,994]
for v in missing_values:
    print 'replacing',v
    X_continuous[X_continuous==v]=999999999

imp = Imputer(missing_values = 999999999, strategy = 'median', axis = 0)
scaler = StandardScaler()
X_continuous = imp.fit_transform(X_continuous)
X_continuous = scaler.fit_transform(X_continuous)

# writing an optimized vw format train:

f = open('data/vw_train.dat','w')

importance = float(len(Y))/sum(Y)

for i in range(len(df)):
    if i%1000 == 0:
        print 'vw formatting :',i,'/',len(df)
    if Y[i]==0:
        label = '-1'
    else:
        label = '1'
    #s = label+ ' ' + str(importance)[0:4] +' '+str(uid[i])+'|'
    s = label+ ' |'
    features = []
    category_row = X_category[i,:]
    features += [c+'_'+str(enc[c][category_row[j]]) for j,c in enumerate(columns_category)]
    continuous_row = X_continuous[i,:]
    features += [c+':'+str(float(continuous_row[j])) for j,c in enumerate(columns_continuous)]
    s += ' '+' '.join(features)+'\n'
    f.write(s)

f.close()




# writing an optimized vw format test:

df = pd.read_csv(ddir+'test.csv',sep=',')
df = df.fillna(999999999)
uid = df.ID.values
df = df.drop(['ID'],axis=1)
X_category = df.as_matrix(columns_category)
X_continuous = df.as_matrix(columns_continuous)
for v in missing_values:
    print 'replacing',v
    X_continuous[X_continuous==v]=999999999

X_continuous = imp.transform(X_continuous)
X_continuous = scaler.transform(X_continuous)
f = open('data/vw_test.dat','w')

for i in range(len(df)):
    if i%1000 == 0:
        print 'vw formatting :',i,'/',len(df)
    label = '-1'
    #s = label+ ' '+str(uid[i])+'|'
    s = label+ ' |'
    features = []
    category_row = X_category[i,:]
    features += [c+'_'+str(enc[c][category_row[j]]) for j,c in enumerate(columns_category) if category_row[j] in enc[c]]
    continuous_row = X_continuous[i,:]
    features += [c+':'+str(float(continuous_row[j])) for j,c in enumerate(columns_continuous)]
    s += ' '+' '.join(features)+'\n'
    f.write(s)

f.close()




# aucs=[]
# for i in range(Xnumeric.shape[1]):
# #for i in range(500):
#     values = np.unique(Xnumeric[:,i])
#     if len(values)>continuous_count_threshold:
#         print values[0],values[-1]
#         xmin = 0
#         xmax = len(values)
#         ymin = values[0]
#         ymax = values[-1]
#         x = np.array(range(0,xmax-xmin),dtype=float)/(xmax-xmin)
#         y = (np.array(values[xmin:xmax],dtype=float) - ymin)/(ymax-ymin)
#         # compute the area under the curve
#         auc = np.sqrt(sum((x-y)**2))/len(x)
#         print auc
#         aucs.append(auc)
#         #if (auc<0.002):
#         plt.plot(x,y)
# 
# plt.show()
# plt.hist(aucs,bins=100)
# plt.show()
