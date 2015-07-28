#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""

from utils import wdir,ddir,header
from utils import training_sample,touch
import pandas as pd
import os
import sys

#######################
#######################
#######################
#######################
# create 
# 10 x balanced data
# for Categorie3
#######################
#######################
#######################
#######################

def create_sample(dftrain,label,mincount,maxsampling,ensemble):
    print '>>> create_sample :',ensemble
    fname = ddir+'training_sampled_'+label+'.csv.'+str(ensemble)
    ftmp = fname+'.tmp'
    if os.path.isfile(fname) or os.path.isfile(fname+'.tmp'):
        return
    touch(ftmp)
    dfsample = training_sample(dftrain,label,mincount,maxsampling)
    dfsample.to_csv(fname,sep=';',index=False,header=False)
    os.remove(ftmp)
    print '<<< create_sample :',ensemble
    return

dftrain = pd.read_csv(ddir+'training_shuffled_normed.csv',sep=';',names = header()).fillna('').reset_index()

for ensemble in range(10):
    create_sample(dftrain,'Categorie3',1000,50,ensemble)
