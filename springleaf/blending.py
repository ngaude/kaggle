import pandas as pd
import numpy as np

ddir = '/home/ngaude/workspace/github/kaggle/springleaf/data/'

df3 = pd.read_csv(ddir+'resultat3.csv',sep=',')
df4 = pd.read_csv(ddir+'resultat4.csv',sep=',')
df5 = pd.read_csv(ddir+'resultat5.csv',sep=',')
df6 = pd.read_csv(ddir+'resultat6.csv',sep=',')
df7 = pd.read_csv(ddir+'resultat7.csv',sep=',')
df8 = pd.read_csv(ddir+'resultat8.csv',sep=',')
df9 = pd.read_csv(ddir+'resultat9.csv',sep=',')
xgb3 = pd.read_csv(ddir+'xgb3.csv',sep=',')
xgb8 = pd.read_csv(ddir+'xgb8.csv',sep=',')
xgbnikko = pd.read_csv(ddir+'xgb_nikko.csv',sep=',')

n = 145232

Y = np.zeros(shape=(n,10),dtype=float)

Y[:,0] = df3.target
Y[:,1] = df4.target
Y[:,2] = df5.target
Y[:,3] = df6.target
Y[:,4] = df7.target
Y[:,5] = df8.target
Y[:,6] = df9.target
Y[:,7] = xgb3.target
Y[:,8] = xgb8.target
Y[:,9] = xgbnikko.target

target = np.median(Y,axis=1)

df = pd.DataFrame(df3)
df.target = target

print 'average target',np.mean(target)

df.to_csv(ddir+'blending.csv',index=False)

low = np.min(Y,axis=1)>0.5
high = np.max(Y,axis=1)>0.5
confidence = (low==high) & (high==True)

##################
# confidence-cooking : best e.g. xgb3 with rounded [0,1] if confidence is true.

target = xgb3.target
import matplotlib.pyplot as plt
plt.hist([target[i] for i in range(n) if confidence[i]==True],bins=100,label='certainty',alpha=0.5)
plt.hist([target[i] for i in range(n) if confidence[i]==False],bins=100,label='doubt',alpha=0.5)
plt.legend()
plt.show()

target = [np.round(target[i]) if confidence[i]==True else target[i] for i in range(n)]

df = pd.DataFrame(df3)
df.target = target

print 'average target',np.mean(target)

df.to_csv(ddir+'xgb3_with_confidence.csv',index=False)





