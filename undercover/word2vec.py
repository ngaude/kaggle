#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils import wdir,ddir,header,normalize_file,add_txt

import pandas as pd
import os
import logging
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

# ****** Define functions to create average word vectors
#



# Read data from files

ext = '.0'

dftrain = pd.read_csv(ddir+'training_random.csv'+ext,sep=';',names = header()).fillna('')
dfvalid = pd.read_csv(ddir+'validation_random.csv'+ext,sep=';',names = header()).fillna('')
dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')

add_txt(dftrain)
add_txt(dfvalid)
add_txt(dftest)

dftrain = dftrain[['Categorie3','Categorie1','txt']]
dfvalid = dfvalid[['Categorie3','Categorie1','txt']]
dftest = dftest[['Identifiant_Produit','txt']]


# ****** Split the labeled and unlabeled training sets into clean sentences
#
sentences = []  # Initialize an empty list of sentences

print "Parsing sentences from training set"
sentences = [t.split() for t in dftrain.txt]
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
min_word_count = 4    # Minimum word count
num_workers = 3       # Number of threads to run in parallel
context = 6           # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print "Training Word2Vec model..."
model = Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling, seed=1)

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
model.most_similar("man")
model.most_similar("queen")
model.most_similar("awful")



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

X = getAvgFeatureVecs( sentences, model, num_features )

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

cla = RandomForestClassifier( n_estimators = 100, n_jobs = 2 , verbose = 1)
#cla = LogisticRegression(C=100)

cla = cla.fit( X, Y1 )
result = cla.score( Xv, Yv1 )
print 'result =',result

