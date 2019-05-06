#!/usr/bin/python3.7

from textblob_de import TextBlobDE
import pandas as pd
import json
import argparse
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# TrainVariables
bodyStr = ''
titleStr = ''
authorStr = ''
categoryStr = ''
firstCategory = ''
allCategoryStr = set()
isbnStr = ''
bookArray = []
data = []
isbnData = []
currPos = 0

# TestVariables
tbodyStr = ''
ttitleStr = ''
tauthorStr = ''
tcategoryStr = ''
tfirstCategory = ''
tallCategoryStr = set()
tisbnStr = ''
tbookArray = []
tdata = []
tisbnData = []

# Other
verbose = False
curr = 0
X_train = None
X_test = None
y_train = None
y_test = None
y_predRF = None
y_predRN = None
y_predGNB = None
y_predMNB = None
y_predBNB = None
genreData = []
dictLU = {}
dictR = {}
dictKJ = {}
dictS = {}
dictGB = {}
dictGE = {}
dictK = {}
dictAG = {}
stopwords = None

X_trainNoDict = None
X_testNoDict = None
y_testNoDict = None
y_trainNoDict = None
y_predRFNoDict = None
y_predRNNoDict = None
y_predGNBNoDict = None
y_predMNBNoDict = None
y_predBNBNoDict = None

# Ensemble
ensembleArray = [[],[],[],[],[]]
ensembleDecision = []

#labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"]

def stopWordListRead():
    global stopwords
    with open('Data/stopwords_german.txt','r') as file:
        stopwords = json.load(file)

def readTrainData():
    global bodyStr
    global titleStr
    global authorStr
    global categoryStr
    global firstCategory
    global allCategoryStr
    global isbnStr
    global bookArray
    global data
    global isbnData
    # Read TrainData
    with open('Data/blurbs_train2.txt','r') as file:
            for line in file:
                    if line.startswith('<book'):
                            bodyStr = ''
                            titleStr = ''
                            authorStr = ''
                            categoryStr = ''
                            firstCategory = ''
                            allCategoryStr = set()
                            isbnStr = ''
                    elif line.startswith('</book>'):
                            bookArray.append((bodyStr,titleStr,authorStr,allCategoryStr,isbnStr,firstCategory))
                    elif line.startswith('<body>'):
                            bodyStr += line
                            bodyStr = bodyStr[:-8]
                            bodyStr = bodyStr[6:]
                            if bodyStr == '':
                                continue
                    elif line.startswith('<title>'):
                            titleStr += line
                            titleStr = titleStr[:-9]
                            titleStr = titleStr[7:]
                    elif line.startswith('<authors>'):
                            authorStr += line
                            authorStr = authorStr[:-11]
                            authorStr = authorStr[9:]
                    elif line.startswith('<topic d="0">'):
                            categoryStr += line
                            categoryStr = categoryStr[:-9]
                            categoryStr = categoryStr[13:]
                            firstCategory = categoryStr
                            allCategoryStr.add(categoryStr)
                            categoryStr = ''
                    elif line.startswith('<isbn>'):
                            isbnStr += line
                            isbnStr = isbnStr[:-8]
                            isbnStr = isbnStr[6:]

def readTestData():
    global tbodySt
    global ttitleStr
    global tauthorStr
    global tcategoryStr
    global tfirstCategory
    global tallCategoryStr
    global tisbnStr
    global tbookArray
    global tdata
    global tisbnData
    # Read TestData
    with open('Data/blurbs_test.txt','r') as file:
            for line in file:
                    if line.startswith('<book'):
                            tbodyStr = ''
                            ttitleStr = ''
                            tauthorStr = ''
                            tcategoryStr = ''
                            tfirstCategory = ''
                            tallCategoryStr = set()
                            tisbnStr = ''
                    elif line.startswith('</book>'):
                            tbookArray.append((tbodyStr,ttitleStr,tauthorStr,tallCategoryStr,tisbnStr,tfirstCategory))
                    elif line.startswith('<body>'):
                            tbodyStr += line
                            tbodyStr = tbodyStr[:-8]
                            tbodyStr = tbodyStr[6:]
                            if tbodyStr == '':
                                continue
                    elif line.startswith('<title>'):
                            ttitleStr += line
                            ttitleStr = ttitleStr[:-9]
                            ttitleStr = ttitleStr[7:]
                    elif line.startswith('<authors>'):
                            tauthorStr += line
                            tauthorStr = tauthorStr[:-11]
                            tauthorStr = tauthorStr[9:]
                    elif line.startswith('<topic d="0">'):
                            tcategoryStr += line
                            tcategoryStr = tcategoryStr[:-9]
                            tcategoryStr = tcategoryStr[13:]
                            tfirstCategory = tcategoryStr
                            tallCategoryStr.add(tcategoryStr)
                            tcategoryStr = ''
                    elif line.startswith('<isbn>'):
                            tisbnStr += line
                            tisbnStr = tisbnStr[:-8]
                            tisbnStr = tisbnStr[6:]

def readDataOneFile():
    global bodyStr
    global titleStr
    global authorStr
    global categoryStr
    global firstCategory
    global allCategoryStr
    global isbnStr
    global bookArray
    global data
    global isbnData
    # Read TrainData
    with open('Data/blurbs_train.txt','r') as file:
            for line in file:
                    if line.startswith('<book'):
                            bodyStr = ''
                            titleStr = ''
                            authorStr = ''
                            categoryStr = ''
                            firstCategory = ''
                            allCategoryStr = set()
                            isbnStr = ''
                    elif line.startswith('</book>'):
                            bookArray.append((bodyStr,titleStr,authorStr,allCategoryStr,isbnStr,firstCategory))
                    elif line.startswith('<body>'):
                            bodyStr += line
                            bodyStr = bodyStr[:-8]
                            bodyStr = bodyStr[6:]
                            if bodyStr == '':
                                continue
                    elif line.startswith('<title>'):
                            titleStr += line
                            titleStr = titleStr[:-9]
                            titleStr = titleStr[7:]
                    elif line.startswith('<authors>'):
                            authorStr += line
                            authorStr = authorStr[:-11]
                            authorStr = authorStr[9:]
                    elif line.startswith('<topic d="0">'):
                            categoryStr += line
                            categoryStr = categoryStr[:-9]
                            categoryStr = categoryStr[13:]
                            firstCategory = categoryStr
                            allCategoryStr.add(categoryStr)
                            categoryStr = ''
                    elif line.startswith('<isbn>'):
                            isbnStr += line
                            isbnStr = isbnStr[:-8]
                            isbnStr = isbnStr[6:]

def splitDataTrainTest():
    global X_train
    global X_test
    global y_train
    global y_test
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7)

def splitData():
    global bookArray
    global tbookArray
    random.shuffle(bookArray)
    tbookArray = bookArray[4000:]
    bookArray = bookArray[:10000]

def loadDictFromFile():
    global dictLU
    global dictR
    global dictKJ
    global dictS
    global dictGB
    global dictGE
    global dictK
    global dictAG
    with open('dictLU.txt','r') as file:
            dictLU = json.load(file)
    with open('dictR.txt','r') as file:
            dictR = json.load(file)
    with open('dictKJ.txt','r') as file:
            dictKJ = json.load(file)
    with open('dictS.txt','r') as file:
            dictS = json.load(file)
    with open('dictGB.txt','r') as file:
            dictGB = json.load(file)
    with open('dictGE.txt','r') as file:
            dictGE = json.load(file)
    with open('dictK.txt','r') as file:
            dictK = json.load(file)
    with open('dictAG.txt','r') as file:
            dictAG = json.load(file)

def addToDict(word):
        global curr
        global dictLU
        global dictR
        global dictKJ
        global dictS
        global dictGB
        global dictGE
        global dictK
        global dictAG
        global stopwords
        if 'Literatur & Unterhaltung' in bookArray[curr][1] and word not in stopwords:
                dictLU[word] = 1
        if 'Ratgeber' in bookArray[curr][1] and word not in stopwords:
                dictR[word] = 1
        if 'Kinderbuch & Jugendbuch' in bookArray[curr][1] and word not in stopwords:
                dictKJ[word] = 1
        if 'Sachbuch' in bookArray[curr][1] and word not in stopwords:
                dictS[word] = 1
        if 'Ganzheitliches Bewusstsein' in bookArray[curr][1] and word not in stopwords:
                dictGB[word] = 1
        if 'Glaube & Ethik' in bookArray[curr][1] and word not in stopwords:
                dictGE[word] = 1
        if 'Künste' in bookArray[curr][1] and word not in stopwords:
                dictK[word] = 1
        if 'Architektur & Garten' in bookArray[curr][1] and word not in stopwords:
                dictAG[word] = 1

def createTempDict():
    global bookArray
    global curr
    for book in bookArray:
            blob = TextBlobDE(book[0])
            textTockens = blob.tags
            for i in textTockens:
                    if i[1] == 'NN' or i[1] == 'NNS' or i[1] == 'NNP' or i[1] == 'NNPS':
                            addToDict(i[0])
                    elif i[1] == 'JJ' or i[1] == 'JJR' or i[1] == 'JJS':
                            addToDict(i[0])
                    elif i[1] == 'VB' or i[1] == 'VBZ' or i[1] == 'VBP' or i[1] == 'VBD' or i[1] == 'VBN' or i[1] == 'VBG':
                            addToDict(i[0])
            curr += 1

def featurize(text):
        j = 0
        k = 0
        nNouns = 0
        nVerbs = 0
        nAdjectives = 0
        nCommas = text.count(',')
        nSym = 0
        tockens = 0
        blob = TextBlobDE(text)
        for sentence in blob.sentences:
                k += 1
                for word in sentence.words:
                        j += 1

        if k == 0:
                k = 1

        j = j / k

        grLU = 0
        grR = 0
        grKJ = 0
        grS = 0
        grGB = 0
        grGE = 0
        grK = 0
        grAG = 0

        textTockens = blob.tags
        for i in textTockens:
                tockens += 1
                if i[1] == 'NN' or i[1] == 'NNS' or i[1] == 'NNP' or i[1] == 'NNPS':
                        nNouns += 1
                elif i[1] == 'JJ' or i[1] == 'JJR' or i[1] == 'JJS':
                        nAdjectives += 1
                elif i[1] == 'VB' or i[1] == 'VBZ' or i[1] == 'VBP' or i[1] == 'VBD' or i[1] == 'VBN' or i[1] == 'VBG':
                        nVerbs += 1
                elif i[1] == 'SYM':
                        nSym += 1

                if i[0] in dictLU:
                        grLU += 1
                if i[0] in dictR:
                        grR += 1
                if i[0] in dictKJ:
                        grKJ += 1
                if i[0] in dictS:
                        grS += 1
                if i[0] in dictGB:
                        grGB += 1
                if i[0] in dictGE:
                        grGE += 1
                if i[0] in dictK:
                        grK += 1
                if i[0] in dictAG:
                        grAG += 1

        if tockens == 0:
                tockens = 1

        rNouns = nNouns / tockens
        rVerbs = nVerbs / tockens
        rAdjectives = nAdjectives / tockens

        allHits = (grLU+grAG+grK+grGE+grGB+grS+grKJ+grR)

        if allHits == 0:
                allHits = 1

        gdrLU = grLU / allHits
        gdrR = grR / allHits
        gdrKJ = grKJ / allHits
        gdrS = grS / allHits
        gdrGB = grGB / allHits
        gdrGE = grGE / allHits
        gdrK = grK / allHits
        gdrAG = grAG / allHits

        nSym += text.count('$')
        nSym += text.count('€')
        nSym += text.count('#')
        nSym += text.count('$')
        nSym += text.count('%')
        nSym += text.count('§')
        nSym += text.count('&')
        nSym += text.count('*')
        nSym += text.count('"')
        nSym += text.count('-')
        nSym += text.count(':')
        nSym += text.count(';')
        
        return j,k,rNouns,rVerbs,rAdjectives,nCommas,nSym,gdrLU,gdrR,gdrKJ,gdrS,gdrGB,gdrGE,gdrK,gdrAG

def createDataArray():
    global currPos
    global bookArray
    global tbookArray
    global data
    global tdata
    global isbnData
    global tisbnData
    # Creation of TrainDataFrame
    for i in bookArray:
            wps,ns,rn,rv,ra,nc,nsym,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG = featurize(bookArray[currPos][0])
            data.append([wps,ns,rn,rv,ra,nc,nsym,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG,bookArray[currPos][5]])
            isbnData.append(bookArray[currPos][4])
            currPos += 1
    # Creation of TestDataFrame
    currPos = 0
    for i in tbookArray:
            wps,ns,rn,rv,ra,nc,nsym,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG = featurize(tbookArray[currPos][0])
            tdata.append([wps,ns,rn,rv,ra,nc,nsym,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG,tbookArray[currPos][5]])
            tisbnData.append(tbookArray[currPos][4])
            currPos += 1

def createDataFrames():
    global data
    global tdata
    global isbnData
    global X_train
    global X_test
    global y_test
    global y_train
    global X_trainNoDict
    global X_testNoDict
    global y_testNoDict
    global y_trainNoDict
    # TrainDataFrame
    dataFrame=pd.DataFrame(data,columns=['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG','Genre'],index=isbnData,dtype=float)

    # TestDataFrame
    tdataFrame=pd.DataFrame(tdata,columns=['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG','Genre'],dtype=float)

    # TrainData
    X_train=dataFrame[['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG']]
    y_train=dataFrame['Genre']

    # TestData
    X_test=tdataFrame[['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG',]]
    y_test=tdataFrame['Genre']

    # TrainDataNoDict
    X_trainNoDict=dataFrame[['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols']]
    y_trainNoDict=dataFrame['Genre']

    # TestDataNoDict
    X_testNoDict=tdataFrame[['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols']]
    y_testNoDict=tdataFrame['Genre']

def createDataCrossVal():
    global currPos
    global bookArray
    global tbookArray
    global data
    global tdata
    global isbnData
    global tisbnData
    global genreData
    for i in bookArray:
            wps,ns,rn,rv,ra,nc,nsym,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG = featurize(bookArray[currPos][0])
            data.append([wps,ns,rn,rv,ra,nc,nsym,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG])
            genreData.append(bookArray[currPos][5])
            isbnData.append(bookArray[currPos][4])
            currPos += 1

def trainClassifier():
    global X_train
    global X_test
    global y_test
    global y_train

    global X_trainNoDict
    global X_testNoDict
    global y_testNoDict
    global y_trainNoDict
    
    global y_predRF
    global y_predRN
    global y_predGNB
    global y_predMNB
    global y_predBNB

    global y_predRFNoDict
    global y_predRNNoDict
    global y_predGNBNoDict
    global y_predMNBNoDict
    global y_predBNBNoDict
    # RandomForestClassifier
    randomForestClassifier=RandomForestClassifier(n_estimators=50,oob_score=True,random_state=0,class_weight="balanced")
    randomForestClassifier.fit(X_train,y_train)

    # RadiusNeighborClassifier
    radiusNeighborClassifier=RadiusNeighborsClassifier(radius=15.0,outlier_label='Literatur & Unterhaltung')
    radiusNeighborClassifier.fit(X_train,y_train)

    # GaussianNaiveBayesClassifier
    gaussianNaiveBayesClassifier=GaussianNB()
    gaussianNaiveBayesClassifier.fit(X_train,y_train)

    # MultinominalNaiveBayesClassifier
    multinominalNaiveBayesClassifier=MultinomialNB(alpha=0.95)
    multinominalNaiveBayesClassifier.fit(X_train,y_train)

    # BernoulliNaiveBayesClassifier
    bernoulliNaiveBayesClassifier=BernoulliNB(alpha=0.95)
    bernoulliNaiveBayesClassifier.fit(X_train,y_train)

    y_predRF=randomForestClassifier.predict(X_test)
    y_predRN=radiusNeighborClassifier.predict(X_test)
    y_predGNB=gaussianNaiveBayesClassifier.predict(X_test)
    y_predMNB=multinominalNaiveBayesClassifier.predict(X_test)
    y_predBNB=bernoulliNaiveBayesClassifier.predict(X_test)

    ####################################################
    # NoDict
    ####################################################
    randomForestClassifierNoDict=RandomForestClassifier(n_estimators=50,oob_score=True,random_state=0,class_weight="balanced")
    randomForestClassifierNoDict.fit(X_trainNoDict,y_trainNoDict)

    # RadiusNeighborClassifier
    radiusNeighborClassifierNoDict=RadiusNeighborsClassifier(radius=15.0,outlier_label='Literatur & Unterhaltung')
    radiusNeighborClassifierNoDict.fit(X_trainNoDict,y_trainNoDict)

    # GaussianNaiveBayesClassifier
    gaussianNaiveBayesClassifierNoDict=GaussianNB()
    gaussianNaiveBayesClassifierNoDict.fit(X_trainNoDict,y_trainNoDict)

    # MultinominalNaiveBayesClassifier
    multinominalNaiveBayesClassifierNoDict=MultinomialNB(alpha=0.95)
    multinominalNaiveBayesClassifierNoDict.fit(X_trainNoDict,y_trainNoDict)

    # BernoulliNaiveBayesClassifier
    bernoulliNaiveBayesClassifierNoDict=BernoulliNB(alpha=0.95)
    bernoulliNaiveBayesClassifierNoDict.fit(X_trainNoDict,y_trainNoDict)

    y_predRFNoDict=randomForestClassifierNoDict.predict(X_testNoDict)
    y_predRNNoDict=radiusNeighborClassifierNoDict.predict(X_testNoDict)
    y_predGNBNoDict=gaussianNaiveBayesClassifierNoDict.predict(X_testNoDict)
    y_predMNBNoDict=multinominalNaiveBayesClassifierNoDict.predict(X_testNoDict)
    y_predBNBNoDict=bernoulliNaiveBayesClassifierNoDict.predict(X_testNoDict)

    if verbose == True:
        print("Accuracy RandomForest:",metrics.accuracy_score(y_test, y_predRF))
        print("Accuracy RadiusNeighbor:",metrics.accuracy_score(y_test, y_predRN))
        print("Accuracy GaussianNaiveBayes:",metrics.accuracy_score(y_test, y_predGNB))
        print("Accuracy MultinominalNaiveBayes:",metrics.accuracy_score(y_test, y_predMNB))
        print("Accuracy BernoulliNaiveBayes:",metrics.accuracy_score(y_test, y_predBNB))

        print("F-Score RandomForest:",metrics.f1_score(y_test, y_predRF,average='micro'))
        print("F-Score RadiusNeighbor:",metrics.f1_score(y_test, y_predRN,average='micro'))
        print("F-Score GaussianNaiveBayes:",metrics.f1_score(y_test, y_predGNB,average='micro'))
        print("F-Score MultinominalNaiveBayes:",metrics.f1_score(y_test, y_predMNB,average='micro'))
        print("F-Score BernoulliNaiveBayes:",metrics.f1_score(y_test, y_predBNB,average='micro'))

        print("ConfusionMatrix RandomForest:\n",metrics.confusion_matrix(y_test, y_predRF,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"]))
        print("ConfusionMatrix RadiusNeighbor:\n",metrics.confusion_matrix(y_test, y_predRN,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"]))
        print("ConfusionMatrix GaussianNaiveBayes:\n",metrics.confusion_matrix(y_test, y_predGNB,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"]))
        print("ConfusionMatrix MultinominalNaiveBayes:\n",metrics.confusion_matrix(y_test, y_predMNB,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"]))
        print("ConfusionMatrix BernoulliNaiveBayes:\n",metrics.confusion_matrix(y_test, y_predBNB,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"]))

def classifierCorssVal():
    global X_train
    global X_test
    global y_test
    global y_train
    # RandomForestClassifier
    randomForestClassifier=RandomForestClassifier(n_estimators=50,oob_score=True,random_state=0,class_weight="balanced")
    crossVal = cross_validate(randomForestClassifier, data, genreData, cv=10,return_train_score=False)
    if verbose == True:
        print(crossVal['test_score'])

def ensemble():
    global ensembleArray
    global ensembleDecision
    global y_test
    global y_predRF
    global y_predRN
    global y_predGNB
    global y_predMNB
    global y_predBNB
    # Add to Ensemble
    for e in y_predRF:
        ensembleArray[0].append(e)
    for e in y_predRN:
        ensembleArray[1].append(e)
    for e in y_predGNB:
        ensembleArray[2].append(e)
    for e in y_predMNB:
        ensembleArray[3].append(e)
    for e in y_predBNB:
        ensembleArray[4].append(e)

    # EnsembleVoting
    i = 0
    while i < len(ensembleArray[0]):
        vdict = {'Literatur & Unterhaltung' : 0, 'Ratgeber' : 0, 'Kinderbuch & Jugendbuch' : 0, 'Sachbuch' : 0, 'Ganzheitliches Bewusstsein' : 0, 'Glaube & Ethik' : 0, 'Künste' : 0, 'Architektur & Garten' : 0} 
        j = 0
        while j < 5:
            if j == 0 or j == 2:
                w = 2
            else:
                w = 1
            if ensembleArray[j][i] == 'Literatur & Unterhaltung':
                vdict['Literatur & Unterhaltung'] += w
            if ensembleArray[j][i] == 'Ratgeber':
                vdict['Ratgeber'] += w
            if ensembleArray[j][i] == 'Kinderbuch & Jugendbuch':
                vdict['Kinderbuch & Jugendbuch'] += w
            if ensembleArray[j][i] == 'Sachbuch':
                vdict['Sachbuch'] += w
            if ensembleArray[j][i] == 'Ganzheitliches Bewusstsein':
                vdict['Ganzheitliches Bewusstsein'] += w
            if ensembleArray[j][i] == 'Glaube & Ethik':
                vdict['Glaube & Ethik'] += w
            if ensembleArray[j][i] == 'Künste':
                vdict['Künste'] += w
            if ensembleArray[j][i] == 'Architektur & Garten':
                vdict['Architektur & Garten'] += w
            j += 1
        ensembleDecision.append(max(vdict, key=vdict.get))
        i += 1

    if verbose == True:
        print("F-Score Ensemble:",metrics.f1_score(y_test, ensembleDecision,average='micro'))
        print("ConfusionMatrix Ensemble:",metrics.confusion_matrix(y_test, ensembleDecision,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"]))

def verboseOutput():
    global y_test
    global y_testNoDict
    global y_predRF
    global predGNB
    global y_predMNB
    global y_predBNB
    global X_trainNoDict
    global X_testNoDict
    global y_testNoDict
    global y_trainNoDict
    global y_predRFNoDict
    global y_predRNNoDict
    global y_predGNBNoDict
    global y_predMNBNoDict
    global y_predBNBNoDict
    with open("verboseResults.txt","w") as file:
        file.write("F-Score RandomForest:" + str(metrics.f1_score(y_test, y_predRF,average='micro')) + str("\n"))
        file.write("F-Score RadiusNeighbor:" + str(metrics.f1_score(y_test, y_predRN,average='micro')) + str("\n"))
        file.write("F-Score GaussianNaiveBayes:" + str(metrics.f1_score(y_test, y_predGNB,average='micro')) + str("\n"))
        file.write("F-Score MultinominalNaiveBayes:" + str(metrics.f1_score(y_test, y_predMNB,average='micro')) + str("\n"))
        file.write("F-Score BernoulliNaiveBayes:" + str(metrics.f1_score(y_test, y_predBNB,average='micro')) + str("\n"))
        file.write("F-Score Ensemble:" + str(metrics.f1_score(y_test, ensembleDecision,average='micro')) + str("\n"))
        file.write("ConfusionMatrix RandomForest:\n" + str(metrics.confusion_matrix(y_test, y_predRF,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"])) + str("\n"))
        file.write("ConfusionMatrix RadiusNeighbor:\n" + str(metrics.confusion_matrix(y_test, y_predRN,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"])) + str("\n"))
        file.write("ConfusionMatrix GaussianNaiveBayes:\n" + str(metrics.confusion_matrix(y_test, y_predGNB,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"])) + str("\n"))
        file.write("ConfusionMatrix MultinominalNaiveBayes:\n" + str(metrics.confusion_matrix(y_test, y_predMNB,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"])) + str("\n"))
        file.write("ConfusionMatrix BernoulliNaiveBayes:\n" + str(metrics.confusion_matrix(y_test, y_predBNB,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"])) + str("\n"))
        file.write("ConfusionMatrix Ensemble:\n" + str(metrics.confusion_matrix(y_test, ensembleDecision,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"])) + str("\n"))

        file.write("F-Score RandomForestNoDict:" + str(metrics.f1_score(y_testNoDict, y_predRFNoDict,average='micro')) + str("\n"))
        file.write("F-Score RadiusNeighborNoDict:" + str(metrics.f1_score(y_testNoDict, y_predRNNoDict,average='micro')) + str("\n"))
        file.write("F-Score GaussianNaiveBayesNoDict:" + str(metrics.f1_score(y_testNoDict, y_predGNBNoDict,average='micro')) + str("\n"))
        file.write("F-Score MultinominalNaiveBayesNoDict:" + str(metrics.f1_score(y_testNoDict, y_predMNBNoDict,average='micro')) + str("\n"))
        file.write("F-Score BernoulliNaiveBayesNoDict:" + str(metrics.f1_score(y_testNoDict, y_predBNBNoDict,average='micro')) + str("\n"))
        file.write("ConfusionMatrix RandomForestNoDict:\n" + str(metrics.confusion_matrix(y_testNoDict, y_predRFNoDict,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"])) + str("\n"))
        file.write("ConfusionMatrix RadiusNeighborNoDict:\n" + str(metrics.confusion_matrix(y_testNoDict, y_predRNNoDict,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"])) + str("\n"))
        file.write("ConfusionMatrix GaussianNaiveBayesNoDict:\n" + str(metrics.confusion_matrix(y_testNoDict, y_predGNBNoDict,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"])) + str("\n"))
        file.write("ConfusionMatrix MultinominalNaiveBayesNoDict:\n" + str(metrics.confusion_matrix(y_testNoDict, y_predMNBNoDict,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"])) + str("\n"))
        file.write("ConfusionMatrix BernoulliNaiveBayesNoDict:\n" + str(metrics.confusion_matrix(y_testNoDict, y_predBNBNoDict,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"])) + str("\n"))

parser = argparse.ArgumentParser(description='sptm')
parser.add_argument("-v", help="verbose", action="store_true")
parser.add_argument("-f", help="fast test 1 run with dictfromFile",action="store_true")
parser.add_argument("-x", help="n crossvalidation", action="store_true")
parser.add_argument("-n", "--num", help="number of validations")
parser.add_argument("-do", "--dataOutput", help="outputfile")
parser.add_argument("-tr", "--traindata", help="traindatafile")
parser.add_argument("-ts", "--testdata", help="testdatafile")
parser.add_argument("-cv", help="10 crossvalidation", action="store_true")
arg = vars(parser.parse_args())
args = parser.parse_args()

if args.v:
    verbose = True
if args.f:
    readTrainData()
    readTestData()
    loadDictFromFile()
    createDataArray()
    createDataFrames()
    trainClassifier()
    verboseOutput()
if args.x:
    readDataOneFile()
    stopWordListRead()
    runNum = 0
    runRange = int(arg["num"])
    for runNum in range(runRange):
        print("Starting Run: " + str(runNum + 1))
        splitData()
        createTempDict()
        createDataArray()
        createDataFrames()
        trainClassifier()
        ensemble()
        verboseOutput()
if args.cv:
    readDataOneFile()
    stopWordListRead()
    splitData()
    createTempDict()
    createDataCrossVal()
    classifierCorssVal()
    verboseOutput()