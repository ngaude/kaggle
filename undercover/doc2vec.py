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


ext = '.0'

dftrain = pd.read_csv(ddir+'training_random.csv'+ext,sep=';',names = header()).fillna('')

# dfvalid = pd.read_csv(ddir+'validation_random.csv'+ext,sep=';',names = header()).fillna('')
# dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')
# add_txt(dfvalid)
# add_txt(dftest)
# dfvalid = dfvalid[['Categorie3','Categorie1','txt']]
# dftest = dftest[['Identifiant_Produit','txt']]

print "Parsing docs from training set"
docs = [ gensim.models.doc2vec.TaggedDocument(t.split(),[i,]) for i,t in enumerate(get_txt(dftrain)) ]
Y1 = dftrain.Categorie1
Y3 = dftrain.Categorie3


# ****** Set parameters and train the word2vec model
#
# Import the built-in logging module and configure it so that Word2Vec
# creates nice output messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 3    # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 6           # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print "Training Word2Vec model..."
model = Doc2Vec(dm=1, dm_concat=1, size=num_features, window=context, negative=5, hs=0, min_count= min_word_count,sample = downsampling, workers= num_workers)

model.build_vocab(docs)

model.train(docs)

model.infer_vector('anticern corrig imperfect innov'.split())


# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_4minwords_6context"
model.save(ddir+model_name)

model.doesnt_match("man woman child kitchen".split())
model.doesnt_match("france england germany berlin".split())
model.doesnt_match("paris berlin london austria".split())
model.most_similar("queen")
model.most_similar("beig")


def makeFeatureVec(words, model, num_features, vocab):
    words = [w for w in words if w in vocab]
    featureVec = np.mean([model[w] for w in words],axis=0)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    vocab  = set(model.index2word)
    print 'yo'
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model,num_features,vocab)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs



# ****** Create average vectors for the training and test sets
#
print "Creating average feature vecs for training reviews"

X = getAvgFeatureVecs( docs, model, num_features )

print "Creating average feature vecs for test reviews"

vsentences = [t.split() for t in dfvalid.txt]
Yv1 = dfvalid.Categorie1
Yv3 = dfvalid.Categorie3
Xv = getAvgFeatureVecs( vsentences, model, num_features )


joblib.dump((X,Y1,Y3,Xv,Yv1,Yv3),'/tmp/XY')

model = Word2Vec.load(ddir+"300features_4minwords_6context")
(X,Y1,Y3,Xv,Yv1,Yv3) = joblib.load('/tmp/XY')

# filter non classified samples:
rows = np.isnan(X.sum(axis=1))
X = X[~rows]
Y1 = Y1[~rows]
Y3 = Y3[~rows]

rows = np.isnan(Xv.sum(axis=1))
Xv = Xv[~rows]
Yv1 = Yv1[~rows]
Yv3 = Yv3[~rows]


X=X[:500000]
Y1=Y1[:500000]

#cla = RandomForestClassifier( n_estimators = 100, n_jobs = 3 , verbose = 1)
cla = LogisticRegression(C=100)

cla = cla.fit( X, Y1 )
result = cla.score( Xv, Yv1 )
print 'result =',result

