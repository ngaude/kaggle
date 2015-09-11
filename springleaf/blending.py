import pandas as pd
import numpy as np

ddir = '/home/ngaude/workspace/github/kaggle/springleaf/data/'

files = [
'xgb_nikko_10.csv',
'xgb_nikko_12.csv',
'xgb_nikko_13.csv',
'xgb_nikko_14.csv',
'xgb_nikko_15.csv',
'xgb_nikko_6.csv',
'xgb_nikko_9.csv']

dfs = [ pd.read_csv(ddir+f) for f in files]

n = len(dfs[0])
m = len(dfs)

Y = np.zeros(shape=(n,m),dtype=float)

for i in range(m):
    Y[:,i] = dfs[i].target

target = np.median(Y,axis=1)
target = np.mean(Y,axis=1)

df = pd.DataFrame(dfs[0])
df.target = target

print 'average target:',np.mean(target)

df.to_csv(ddir+'xgb_blending.csv',index=False)

low = np.min(Y,axis=1)>0.5
high = np.max(Y,axis=1)>0.5
confidence = (low==high) #& (high==True)

##################
# confidence-cooking : best e.g. xgb3 with rounded [0,1] if confidence is true.

target = df.target
import matplotlib.pyplot as plt
plt.hist([target[i] for i in range(n) if confidence[i]==True],bins=100,label='certainty',alpha=0.5)
plt.hist([target[i] for i in range(n) if confidence[i]==False],bins=100,label='doubt',alpha=0.5)
plt.legend()
plt.show()

target = [np.round(target[i]) if confidence[i]==True else target[i] for i in range(n)]

df.target = target

print 'average target with confidence:',np.mean(target)

df.to_csv(ddir+'xgb_blending_with_confidence.csv',index=False)





