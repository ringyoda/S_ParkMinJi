from numpy import *
import operator
from os import listdir

from numpy import *

def textParse(bigString):    #input is big string, #output is word list
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
        #else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones() 
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
    
def testingNB(text):
    listPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainMat=[]
    for postinDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))

    testEntry = textParse(text)
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    return classifyNB(thisDoc,p0V,p1V,pAb)

def testData():
    testList = []; testClass = []; testCount = 0
    with open('data/metro/MetroTestSet.txt') as mTest:
        for i, line in enumerate(mTest):
            currLine = line.strip().split('\t')
            testClass.extend(currLine[0])
            testList.append(currLine[1])
            testCount += 1         
    errorCount = 0.0
    for i in range(0, testCount):
        if testingNB(testList[i]) != int(testClass[i]):
            errorCount += 1
        print 'origin class: ', int(testClass[i]), '  classified as: ', testingNB(testList[i])
    print 'the error rate is:', errorCount/testCount