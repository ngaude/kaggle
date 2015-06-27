# -*- coding: utf-8 -*-

""" Hands on Kaggle titanic challenge https://www.kaggle.com/c/titanic/
Author : ngaude
Date : 27 June 2015
please see https://github.com/ngaude
""" 
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import cross_validation
import matplotlib.pyplot as plt



############################################################
fpath = './'
#fpath = 'C:/Users/ngaude/Downloads/'
#fpath = '/home/ngaude/Downloads/'
#fpath = 'C:/Users/Utilisateur/Downloads/'
############################################################

#translate(None,['"'()']replace('mrs').replace('mr').replace('miss')

import unicodedata
import re
#from ngram import NGram



def parse_name(s):
    s = s.replace('Ø','O')
    s = s.replace('ø','o')
    s = s.translate(None,'_\'."()-').lower()
    s = unicode(s,'utf-8')
    s = ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))
    s = s.replace('mrs','').replace('miss','').replace('mr','')
    s = s.replace('melle','').replace('jr','')
    s = s.replace('master','').replace('colonel','')
    s = s.replace('dr','').replace('countess of','')
    s = re.sub(' +',' ',s)
    return s

m = pd.read_csv(fpath+'victims.csv',skiprows = 1)
n = m['name_link/_text']
victims = {}
for i in n.map(parse_name):
    family_name = i.split(',')[0].replace(' ','')
    first_name = i.split(',')[1].strip().split(' ')
    victims[family_name] = victims.get(family_name, []) + first_name


m = pd.read_csv(fpath+'survivors.csv',skiprows = 1)
n = m['name_link/_text']
survivors = {}
for i in n.map(parse_name):
    family_name = i.split(',')[0].replace(' ','')
    first_name = i.split(',')[1].strip().split(' ')
    survivors[family_name] = survivors.get(family_name, []) + first_name
    

def is_survivor(s):
    # returns True if survivor
    # returns False if victim
    # return None if do not know
    s = parse_name(s)
    family_name = s.split(',')[0].replace(' ','')
    first_name = s.split(',')[1].strip().split(' ')
    
    survivor = None
    if family_name in survivors:
        fname = survivors[family_name]
        for i in first_name:
            if i in fname:
                survivor = True
    if family_name in victims:
        fname = victims[family_name]
        for i in first_name:
            if i in fname:
                if survivor == True:
                    # conflict victims or survivor
                    survivor = 666
                else:
                    survivor = False
                return survivor
    return survivor
    

train_df = pd.read_csv(fpath+'train.csv', header=0)  
train_df['is_survivor'] = train_df.Name.map(is_survivor)

error =  train_df[train_df['Survived'] != train_df['is_survivor']]

                


#survivors_df = pd.read_csv(fpath+'survivors.csv')
#s_name = survivors_df['name_link/_text']
#train_df = pd.read_csv(fpath+'train.csv', header=0)  
#t_name = train_df.Name

#
## Data cleanup
## TRAIN DATA
#train_df = pd.read_csv(fpath+'train.csv', header=0)        # Load the train file into a dataframe
#
## I need to convert all strings to integer classifiers.
## I need to fill in the missing values of the data and make it complete.
#
## female = 0, Male = 1
#train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
#
## Embarked from 'C', 'Q', 'S'
## Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.
#
## All missing Embarked -> just make them embark from most common place
#if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
#    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values
#
#Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
#Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
#train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int
#
## All the ages with no data -> make the median of all Ages
#median_age = train_df['Age'].dropna().median()
#if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
#    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age
#
## Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
#train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
#
#
## TEST DATA
#test_df = pd.read_csv(fpath+'test.csv', header=0)        # Load the test file into a dataframe
#
## I need to do the same with the test data now, so that the columns are the same as the training data
## I need to convert all strings to integer classifiers:
## female = 0, Male = 1
#test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
#
## Embarked from 'C', 'Q', 'S'
## All missing Embarked -> just make them embark from most common place
#if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
#    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
## Again convert all Embarked strings to int
#test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)

#
## All the ages with no data -> make the median of all Ages
#median_age = test_df['Age'].dropna().median()
#if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
#    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age
#    
#    
#
## All the missing Fares -> assume median of their respective class
#if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
#    median_fare = np.zeros(3)
#    for f in range(0,3):                                              # loop 0 to 2
#        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
#    for f in range(0,3):                                              # loop 0 to 2
#        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]
#
## Collect the test data's PassengerIds before dropping it
#ids = test_df['PassengerId'].values
## Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
#test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
#
#
## The data is now ready to go. So lets fit to the train, then predict to the test!
## Convert back to a numpy array
#train_data = train_df.values
#test_data = test_df.values
#
##
##cv_nscore = np.zeros(100)
##cv_score = np.zeros(10)
##for i in range(4,100):
##    for j in range(10):
##        xa,xb,ya,yb = cross_validation.train_test_split(train_data[0::,1::],train_data[0::,0],train_size=0.8)
##        forest = RandomForestClassifier(n_estimators=i)
##        forest = forest.fit( xa,ya)
##        cv_score[j] = forest.score(xb,yb)
##    cv_nscore[i] = np.median(cv_score)
##    print 'CrossValidation Score(',i,'):',cv_nscore[i]
##
##
##
##mortality = 1502./2224
##
##for i in range(100):
##    xa,xb,ya,yb = cross_validation.train_test_split(train_data[0::,1::],train_data[0::,0],train_size=0.8)
##    forest = RandomForestClassifier(n_estimators=2000) #,class_weight = {0:mortality,1:1-mortality})
##    forest = forest.fit( xa,ya)
##    print 'score',forest.score(xb,yb)
#
#xa,xb,ya,yb = cross_validation.train_test_split(train_data[0::,1::],train_data[0::,0],train_size=0.8)
#ada = AdaBoostClassifier(n_estimators=500) #,class_weight = {0:mortality,1:1-mortality})
#ada = forest.fit( xa,ya)
#print 'score',ada.score(xb,yb)
#
#
#
#print 'Training...'
#forest = RandomForestClassifier(n_estimators=200)
#forest = forest.fit( train_data[0::,1::], train_data[0::,0] )
#
#print 'Predicting...'
#output = forest.predict(test_data).astype(int)
#
#
#predictions_file = open(fpath+"myfirstforest.csv", "wb")
#open_file_object = csv.writer(predictions_file)
#open_file_object.writerow(["PassengerId","Survived"])
#open_file_object.writerows(zip(ids, output))
#predictions_file.close()
#print 'Done.'