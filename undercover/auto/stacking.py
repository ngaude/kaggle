#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: ngaude
"""

import pandas as pd
import numpy as np
from utils import itocat3,cat3tocat1,cat1toi
from sklearn.externals import joblib

ddir = '/home/ngaude/workspace/data/cdiscount.auto/' 

stage3_proba_stack = np.full(shape=(35065,5789),fill_value = 0.,dtype = float)
stage1_proba_stack = np.full(shape=(35065,52),fill_value = 0.,dtype = float)

#exts = range(300,312)
#exts = range(100,112)
exts = range(100,112)+range(16,21)
#exts = range(16,20)


def bayes_prediction(stage1_log_proba,stage3_log_proba):
    for i in range(stage3_log_proba.shape[1]):
        cat3 = itocat3[i]
        cat1 = cat3tocat1[cat3]
        j = cat1toi[cat1]
        stage3_log_proba[:,i] += stage1_log_proba[:,j]

for i in exts:
    ext = '.'+str(i)
    print '>> stacking ',ext,'/',len(exts)
    if (i>=200) and (i<300):
        # FIXME : sounds like 2xx results (danya) are not in good shape !!!
        stage1_log_proba_test = np.load(ddir+'joblib/log_proba_test'+ext+'_01.npy')
        stage3_log_proba_test = np.load(ddir+'joblib/log_proba_test'+ext+'_02.npy')
    else:
        (stage1_log_proba_test,stage3_log_proba_test) = joblib.load(ddir+'/joblib/log_proba_test'+ext)
    bayes_prediction(stage1_log_proba_test,stage3_log_proba_test)
    stage1_proba_stack += np.exp(stage1_log_proba_test)
    stage3_proba_stack += np.exp(stage3_log_proba_test)
    del stage1_log_proba_test,stage3_log_proba_test

joblib.dump((exts,stage1_proba_stack,stage3_proba_stack),'proba_test_stacked.'+str(min(exts))+'-'+str(max(exts)))

# joblib.dump((exts,stage1_proba_stack,stage3_proba_stack),'proba_test_stacked.100-116')

