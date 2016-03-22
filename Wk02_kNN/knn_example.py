
from numpy import *

def classify0(inX, dataSet, labels, k):
    dataSet = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]) #array
    group = ['A','A','B','B']
    inX = [0,0] #[]괄호모양 list를 의미, inX = list()와 동일
    k = 3

    dataSetSize = dataSet.shape[0] # dataSet 행렬의 row 갯수
    diffMat = tile(inX, (dataSetSize,1)) - dataSet # dataSet과 inX의 차를 계산
    sqDiffMat = diffMat**2 # 제곱 연산
    sqDistances = sqDiffMat.sum(axis=1) # 두 데이터의 합을 계산
    distances = sqDistances**0.5 # 루트 연산

    sortedDistIndicies = distances.argsort() # 오름차순 정렬 ~ array type이므로 해당 함수를 쓸 수 있음
    classCount = {} #{}괄호모양으로 data type이 dictionary라는 것을 알 수 있음 classCount = dict()과 동일

    for i in range (k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 # a = a + 1
        #print voteIlabel, classCount[voteIlabel]
    
    #print classCount

    #classCount.iteritems()은 Iterator Pattern과 비슷한 기능
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse=True) # 내림차순 정렬

    return sortedClassCount[0][0]