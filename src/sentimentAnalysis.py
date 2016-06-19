'''
Created on June 19, 2016
Reference: NaiveBayesian for Machine Learning in Action Ch. 4
@author: Minji Park
'''

from numpy import *

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def loadDataSet():
    postingList = [];
    with open('data/metro/MetroPositiveTraining.txt') as mPosTrain:
        for i, line in enumerate(mPosTrain):
            postingList.append(textParse(line))
    classVec = list(zeros(150))    #0: Positive
    with open('data/metro/MetroNegativeTraining.txt') as mNegTrain:
        for i, line in enumerate(mNegTrain):
            postingList.append(textParse(line))           
    classVec.extend(ones(150))    #1: Negative
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets - or
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
#         else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec

