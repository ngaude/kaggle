import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


ddir = '/home/ngaude/workspace/github/kaggle/springleaf/data/'

df = pd.read_csv(ddir+'train.csv',sep=',')
df = df.fillna(999999999)

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
plt.plot(np.log(np.array([np.percentile(a,i) for i in range(100)])))
plt.show()
# we decide to cut the numeric categorical vs discrete threshold at 68 percentile.
discrete_count_threshold = int(np.percentile(a,68))
# e.g. 101

assert discrete_count_threshold == 101

columns_numeric = columns_int+columns_float

columns_category = [c for c in columns_numeric if df[c].nunique() <= discrete_count_threshold]

columns_discrete = [c for c in columns_numeric if df[c].nunique() > discrete_count_threshold]

print 'columns_category =',len(columns_category)
print 'columns_discrete =',len(columns_discrete)

# one hot encoding of columns_category
# TODO


# linear or log discretization + standardization + variance normalization...
# TODO

from sklearn.preprocessing import Imputer
Xnumeric = df.as_matrix(columns_discrete)
missing_values = [999999994,999999995,999999996,999999997,999999998,999999999,-99999,999999,9999,9998,9994,9995,9996,9997,9998,999,998,997,996,995,994]
for v in missing_values:
    print 'replacing',v
    Xnumeric[Xnumeric==v]=999999999

imp = Imputer(missing_values = 999999999, strategy = 'median', axis = 0)

Xnumeric = imp.fit_transform(Xnumeric)


aucs=[]
for i in range(Xnumeric.shape[1]):
#for i in range(500):
    values = np.unique(Xnumeric[:,i])
    if len(values)>discrete_count_threshold:
        print values[0],values[-1]
        xmin = 0
        xmax = len(values)
        ymin = values[0]
        ymax = values[-1]
        x = np.array(range(0,xmax-xmin),dtype=float)/(xmax-xmin)
        y = (np.array(values[xmin:xmax],dtype=float) - ymin)/(ymax-ymin)
        # compute the area under the curve
        auc = np.sqrt(sum((x-y)**2))/len(x)
        print auc
        aucs.append(auc)
        #if (auc<0.002):
        plt.plot(x,y)

plt.show()

plt.hist(aucs,bins=100)
plt.show()
