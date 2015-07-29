import marisa_trie
from sklearn.externals import six
from sklearn.feature_extraction.text import TfidfVectorizer
#import Stemmer # NOTE : pip install pyStemmer
import nltk
from bs4 import BeautifulSoup
import re
import unicodedata 
import time
import pandas as pd
import numpy as np
import random
import os
from sklearn.neighbors import NearestNeighbors
from scipy import sparse


"""
ddir = 'E:/workspace/data/cdiscount/'
wdir = 'C:/Users/ngaude/Documents/GitHub/kaggle/undercover/'
"""
ddir = '/home/ngaude/workspace/data/cdiscount/'
wdir = '/home/ngaude/workspace/github/kaggle/undercover/'

rayon = pd.read_csv(ddir+'rayon.csv',sep=';')

itocat1 = list(np.unique(rayon.Categorie1))
cat1toi = {cat1:i for i,cat1 in enumerate(itocat1)}
itocat2 = list(np.unique(rayon.Categorie2))
cat2toi = {cat2:i for i,cat2 in enumerate(itocat2)}
itocat3 = list(np.unique(rayon.Categorie3))
cat3toi = {cat3:i for i,cat3 in enumerate(itocat3)}
cat3tocat2 = rayon.set_index('Categorie3').Categorie2.to_dict()
cat3tocat1 = rayon.set_index('Categorie3').Categorie1.to_dict()
cat2tocat1 = rayon[['Categorie2','Categorie1']].drop_duplicates().set_index('Categorie2').Categorie1.to_dict()
cat1count = len(np.unique(rayon.Categorie1))
cat2count = len(np.unique(rayon.Categorie2))
cat3count = len(np.unique(rayon.Categorie3))


stopwords = []
with open(wdir+'stop-words_french_1_fr.txt', "r") as f:
    stopwords += f.read().split('\n')

with open(wdir+'stop-words_french_2_fr.txt', "r") as f:
    stopwords += f.read().split('\n')

stopwords += nltk.corpus.stopwords.words('french')
stopwords += ['voir', 'presentation']
stopwords = set(stopwords)

#stemmer = Stemmer.Stemmer('french')
stemmer=nltk.stem.SnowballStemmer('french')

def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

def header(test=False):
    if test==True:
        columns = ['Identifiant_Produit','Description','Libelle','Marque','prix']
    else:
        columns = ['Identifiant_Produit','Categorie1','Categorie2','Categorie3','Description','Libelle','Marque','Produit_Cdiscount','prix']
    return columns

def normalize_txt(txt):
    # remove html stuff
    txt = BeautifulSoup(txt,from_encoding='utf-8').get_text()
    # lower case
    txt = txt.lower()
    # special escaping character '...'
    txt = txt.replace(u'\u2026','.')
    txt = txt.replace(u'\u00a0',' ')
    # remove accent btw
    txt = unicodedata.normalize('NFD', txt).encode('ascii', 'ignore')
    #txt = unidecode(txt)
    # remove non alphanumeric char
    txt = re.sub('[^a-z_]', ' ', txt)
    # remove french stop words
    tokens = [w for w in txt.split() if (len(w)>2) and (w not in stopwords)]
    # french stemming
    tokens = [ stemmer.stem(token) for token in tokens]
#    tokens = stemmer.stemWords(tokens)
    return ' '.join(tokens)

def normalize_price(price):
    if (price<0) or (price>100):
        price = 0
    return price

def normalize_file(fname,header,nrows = None):
    columns = {k:v for v,k in enumerate(header)}
    ofname = fname.split('.')[0]+'_normed.'+fname.split('.')[1]
    ff = open(ofname,'w')
    start_time = time.time()
    counter = 0
    for line in open(fname):
        if line.startswith('Identifiant_Produit'):
            continue
        di = columns['Description']
        li = columns['Libelle']
        mi = columns['Marque']
        pi = columns['prix']
        if counter%1000 == 0:
            print fname,': lines=',counter,'time=',int(time.time() - start_time),'s'
        ls = line.split(';')
        # marque normalization
        txt = ls[mi]
        txt = re.sub('[^a-zA-Z0-9]', '_', txt).lower()
        ls[mi] = txt
        #
        # description normalization
        ls[di] = normalize_txt(ls[di])
        #
        # libelle normalization
        ls[li] = normalize_txt(ls[li])
        #
        # prix normalization
        ls[pi] = str(normalize_price(float(ls[pi].strip())))
        line = ';'.join(ls)
        ff.write(line+'\n')
        counter += 1
        if (nrows is not None) and (counter>=nrows):
            break
    ff.close()
    return

class iterText(object):
    def __init__(self, df):
        """
        Yield each document in turn, as a text.
        """
        self.df = df
    
    def __iter__(self):
        for row_index, row in self.df.iterrows():
            if (row_index>0) and (row_index%10000)==0:
                print row_index,'/',len(self.df)
            txt = ' '.join([row.Marque]*3+[row.Libelle]*2+[row.Description])
            yield txt
    
    def __len__(self):
        return len(self.df)

class MarisaTfidfVectorizer(TfidfVectorizer):
    def fit_transform(self, raw_documents, y=None):
        super(MarisaTfidfVectorizer, self).fit_transform(raw_documents)
        self._freeze_vocabulary()
        return super(MarisaTfidfVectorizer, self).fit_transform(raw_documents, y)
    def fit(self, raw_documents, y=None):
        super(MarisaTfidfVectorizer, self).fit(raw_documents)
        self._freeze_vocabulary()
        return super(MarisaTfidfVectorizer, self).fit(raw_documents, y)
    def _freeze_vocabulary(self, X=None):
        if not self.fixed_vocabulary_:
            self.vocabulary_ = marisa_trie.Trie(six.iterkeys(self.vocabulary_))
            self.fixed_vocabulary_ = True
            del self.stop_words_


def training_sample(dftrain,label,mincount = 200,maxsampling = 10):
    cl = dftrain[label]
    cc = cl.groupby(cl)
    s = (cc.count() > mincount/maxsampling)
    labelmaj = s[s].index
    print len(labelmaj),len(labelmaj)*mincount
    dfs = []
    for i,cat in enumerate(labelmaj):
        if i%10==0:
            print i,'/',len(labelmaj),':'
        df = dftrain[dftrain[label] == cat]
        if len(df)>=mincount:
            # undersample mincount samples
            rows = random.sample(df.index, mincount)
            dfs.append(df.ix[rows])
        else:
            # sample all samples + oversample the remaining
            dfs.append(df)
            df = df.iloc[np.random.randint(0, len(df), size=mincount-len(df))]
            dfs.append(df)
    dfsample = pd.concat(dfs)
    dfsample = dfsample.reset_index(drop=True)
    dfsample = dfsample.reindex(np.random.permutation(dfsample.index),copy=False)
    return dfsample


# @param : X the matrix of vector-space sample
# @param : Y the labels of sample
# @param : minclass the "minority" class to sample using ADASYN algorithm
# @param : K the amount of neighbours to sample for Knn
# @param : n the "approximate" number of sample to be returned
# @return : the synthetic matrix of vector-space sample
# NOTE : if "minority" class sample > n , then returns a random sample without replacement of "minority" class sample e.g. simple undersample behavior

def adasyn_sample(X,Y,minclass,K=5,n=200):
    indices = np.nonzero(Y==minclass)
    Ymin = Y[indices]
    Xmin = X[indices]
    Cmin = len(indices[0])
    Xs = []
    if n > Cmin:
        Xs.append(Xmin)
        n -= len(Ymin)
    else:
        # simple random without replacement undersampling
        return Xmin[random.sample(range(Cmin),n)]
    neigh = NearestNeighbors(n_neighbors=30)
    neigh.fit(X)
    nindices = neigh.kneighbors(Xmin,K,False)
    gamma = [float(sum(Y[i]==minclass))/K for i in nindices]
    gamma = gamma / np.linalg.norm(gamma,ord = 1)
    neigh = NearestNeighbors(n_neighbors=30)
    neigh.fit(Xmin)
    N = np.round(gamma*n).astype(int)
    assert len(N) == Cmin
    for (i,nn) in enumerate(N):
        nindices = neigh.kneighbors(Xmin[i],K,False)[0]
        for j in range(nn):
            alpha = random.random()
            Xnn = X[random.choice(nindices)]
            Xs.append((1.-alpha)*Xmin[i]+alpha*Xnn)
    Xadasyn = sparse.vstack(Xs)  
    return Xadasyn

def add_txt(df):
    assert 'Marque' in df.columns
    assert 'Libelle' in df.columns
    assert 'Description' in df.columns
    assert 'prix' in df.columns
    df['txt'] = 'px'+(np.log2(df.prix+1)).astype(int).astype(str)+' '+(df.Marque+' ')*3+(df.Libelle+' ')*2+df.Description
    return
