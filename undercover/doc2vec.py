#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils import wdir,ddir,header,normalize_file,add_txt,get_txt

import pandas as pd
import os
import logging
import numpy as np
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import gensim
import random


ext = '.0'

dftrain = pd.read_csv(ddir+'training_random.csv'+ext,sep=';',names = header()).fillna('')
dftrain = dftrain.drop_duplicates()
dfvalid = pd.read_csv(ddir+'validation_random.csv'+ext,sep=';',names = header()).fillna('')
dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')

num_train = len(dftrain)
num_valid = len(dfvalid)
num_test  = len(dftest)

txts = np.concatenate((get_txt(dftrain),get_txt(dfvalid),get_txt(dftest)))
Y1  = np.concatenate((dftrain.Categorie1.values,dfvalid.Categorie1.values))
Y3  = np.concatenate((dftrain.Categorie3.values,dfvalid.Categorie3.values))

print "Parsing docs from training set"
docs = [ gensim.models.doc2vec.TaggedDocument(t.split(),['_'+str(i),]) for i,t in enumerate(txts) ]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 3    # Minimum word count
num_workers = 3       # Number of threads to run in parallel
context = 6           # Context window size
downsampling = 1e-5   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print "Training Word2Vec model..."
model = Doc2Vec(dm=1, dm_concat=1, size=num_features, window=context, hs=1, min_count= min_word_count,sample = downsampling, workers= num_workers)

model.build_vocab(docs)

#for i in range(9):
#    model.train(docs)
#    random.shuffle(docs)
#
#model.infer_vector('anticern corrig imperfect innov'.split())


# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

##################
# sanity check
##################
doc_id = 1234
docs[doc_id]
assert int(docs[doc_id].tags[0][1:]) == doc_id
most_similar_doc = model.docvecs.most_similar(positive=['_'+str(doc_id)])[0]
most_similar_doc_id = int(most_similar_doc[0][1:])
docs[most_similar_doc_id]
model.most_similar('parent')
model.most_similar('maman')
model.doesnt_match("pari marseil lyon voitur".split())
###################

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_3minwords_6context.doc2vec"
model.save(ddir+model_name)

X = np.zeros((len(model.docvecs),num_features),dtype="float32")
for i,x in enumerate(model.docvecs):
    X[i,:] =x

joblib.dump((X,Y1,Y3),ddir+'joblib/doc2vecsXY')

#  model = Word2Vec.load(ddir+"300features_4minwords_6context")
(X,Y1,Y3,Xv,Yv1,Yv3) = joblib.load('/tmp/XY')

# filter non classified samples:
rows = np.isnan(X.sum(axis=1))
Xt = X[~rows]
Yt = Y1[~rows]
Y3 = Y3[~rows]

rows = np.isnan(Xv.sum(axis=1))
Xv = Xv[~rows]
Yv1 = Yv1[~rows]
Yv3 = Yv3[~rows]


Xt=X[:num_train]
Y1t=Y1t[:num_train]

cla = RandomForestClassifier( n_estimators = 100, n_jobs = 3 , verbose = 1)

cla = cla.fit( X, Y1 )
result = cla.score( Xv, Yv1 )
print 'result =',result

