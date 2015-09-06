import pandas as pd
import sys
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt

f = sys.argv[1]

ddir = '/home/ngaude/workspace/github/kaggle/springleaf/data/'
cv = pd.read_csv(ddir+'cv.csv',low_memory=False)
pred = pd.read_csv(f)
assert len(cv) == len(pred)
assert sum(cv.ID.values != pred.ID.values)==0

score = roc_auc_score(cv.target,pred.target)
print '*'*50
print 'score =',score
print '*'*50

fpr, tpr, _ = roc_curve(cv.target, pred.target)

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (score = %0.4f)' % score)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show(block=False)

plt.figure()
plt.title('error spread')
plt.hist(np.abs(cv.target- pred.target),bins=100)
plt.show(block=False)

raw_input("Press Enter to continue...")


