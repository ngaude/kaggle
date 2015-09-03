import pandas as pd
import numpy as np

ddir = '/home/ngaude/workspace/github/kaggle/springleaf/data/'

df3 = pd.read_csv(ddir+'resultat3.csv',sep=',')
df4 = pd.read_csv(ddir+'resultat4.csv',sep=',')
df5 = pd.read_csv(ddir+'resultat5.csv',sep=',')
df6 = pd.read_csv(ddir+'resultat6.csv',sep=',')
df7 = pd.read_csv(ddir+'resultat7.csv',sep=',')

n = 145232

Y = np.zeros(shape=(n,5),dtype=float)

Y[:,0] = df3.target
Y[:,1] = df4.target
Y[:,2] = df5.target
Y[:,3] = df6.target
Y[:,4] = df7.target

target = np.median(Y,axis=1)

df = pd.DataFrame(df3)
df.target = target

print 'average target',np.mean(target)

df.to_csv('data/vw_submission.csv',index=False)
