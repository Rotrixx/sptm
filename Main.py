#!/usr/bin/python3.7

from textblob_de import TextBlobDE
import pandas as pd
import numpy as np
import json
import argparse
import random
import timeit
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.tree import export_graphviz

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

allWords = set()

# Other
verbose = False
curr = 0
X_train = None
X_test = None
y_train = None
y_test = None
y_predRF = None
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
estimator = None

dataVec = []
tdataVec = []

trainTexts = []
testTexts = []

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
                            for i in allCategoryStr:
                                if bodyStr == '':
                                    continue
                                if isbnStr.startswith('4'):
                                    continue
                                bookArray.append((bodyStr,titleStr,authorStr,allCategoryStr,isbnStr,i))
                            """
                            if bodyStr == '':
                                    continue
                            if isbnStr.startswith('4'):
                                    continue
                            bookArray.append((bodyStr,titleStr,authorStr,allCategoryStr,isbnStr,firstCategory))
                            """
                    elif line.startswith('<body>'):
                            bodyStr += line
                            bodyStr = bodyStr[:-8]
                            bodyStr = bodyStr[6:]
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
                    elif line.startswith('<topic d="0" label'):
                            categoryStr += line
                            categoryStr = categoryStr[:-9]
                            categoryStr = categoryStr[26:]
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

def splitter(array, size):
    helperArray = []
    while len(array) > size:
        pice = array[:size]
        helperArray.append(pice)
        array = array[size:]
    helperArray.append(array)
    return helperArray

def splitData():
    global bookArray
    global tbookArray
    random.shuffle(bookArray)
    helper = splitter(bookArray,10000)
    tbookArray = helper[1]
    bookArray = helper[0]
    print(len(tbookArray))
    print(len(bookArray))

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
        global allWords

        if 'Literatur & Unterhaltung' in bookArray[curr][5] and word not in stopwords:
                if word in dictLU:
                        dictLU[word] += 1
                else:
                        dictLU[word] = 1
        if 'Ratgeber' in bookArray[curr][5] and word not in stopwords:
                if word in dictR:
                        dictR[word] += 1
                else:
                        dictR[word] = 1
        if 'Kinderbuch & Jugendbuch' in bookArray[curr][5] and word not in stopwords:
                if word in dictKJ:
                        dictKJ[word] += 1
                else:
                        dictKJ[word] = 1
        if 'Sachbuch' in bookArray[curr][5] and word not in stopwords:
                if word in dictS:
                        dictS[word] += 1
                else:
                        dictS[word] = 1
        if 'Ganzheitliches Bewusstsein' in bookArray[curr][5] and word not in stopwords:
                if word in dictGB:
                        dictGB[word] += 1
                else:
                        dictGB[word] = 1
        if 'Glaube & Ethik' in bookArray[curr][5] and word not in stopwords:
                if word in dictGE:
                        dictGE[word] += 1
                else:
                        dictGE[word] = 1
        if 'Künste' in bookArray[curr][5] and word not in stopwords:
                if word in dictK:
                        dictK[word] += 1
                else:
                        dictK[word] = 1
        if 'Architektur & Garten' in bookArray[curr][5] and word not in stopwords:
                if word in dictAG:
                        dictAG[word] += 1
                else:
                        dictAG[word] = 1
        allWords.add(word)

def improveDict():
    global dictLU
    global dictR
    global dictKJ
    global dictS
    global dictGB
    global dictGE
    global dictK
    global dictAG
    global allWords

    for word in allWords:
        counter = 0
        if word in dictLU:
            counter += 1
        if word in dictR:
            counter += 1
        if word in dictKJ:
            counter += 1
        if word in dictS:
            counter += 1
        if word in dictGB:
            counter += 1
        if word in dictGE:
            counter += 1
        if word in dictK:
            counter += 1
        if word in dictAG:
            counter += 1

        if word in dictLU:
            dictLU[word] /= counter
        if word in dictR:
            dictR[word] /= counter
        if word in dictKJ:
            dictKJ[word] /= counter
        if word in dictS:
            dictS[word] /= counter
        if word in dictGB:
            dictGB[word] /= counter
        if word in dictGE:
            dictGE[word] /= counter
        if word in dictK:
            dictK[word] /= counter
        if word in dictAG:
            dictAG[word] /= counter

def createTempDict():
    global bookArray
    global curr
    for book in bookArray:
            blob = TextBlobDE(book[0])
            textTockens = blob.tags
            for i in textTockens:
                    if i[1] == 'NN' or i[1] == 'NNS' or i[1] == 'NNP' or i[1] == 'NNPS':
                            addToDict(i[0].lower())
                    elif i[1] == 'JJ' or i[1] == 'JJR' or i[1] == 'JJS':
                            addToDict(i[0].lower())
                    elif i[1] == 'VB' or i[1] == 'VBZ' or i[1] == 'VBP' or i[1] == 'VBD' or i[1] == 'VBN' or i[1] == 'VBG':
                            addToDict(i[0].lower())
            curr += 1

def permutateArray(array):
    arrayLU = []
    arrayR = []
    arrayKJ = []
    arrayS = []
    arrayGB = []
    arrayGE = []
    arrayK = []
    arrayAG = []
    #retArray = []

    i = 0
    print(array[0][5])
    for _ in array:
        if array[i][5] == 'Literatur & Unterhaltung':
            arrayLU.append(array[i])    
        if array[i][5] == 'Ratgeber':
            arrayR.append(array[i])  
        if array[i][5] == 'Kinderbuch & Jugendbuch':
            arrayKJ.append(array[i])  
        if array[i][5] == 'Sachbuch':
            arrayS.append(array[i])  
        if array[i][5] == 'Ganzheitliches Bewusstsein':
            arrayGB.append(array[i])  
        if array[i][5] == 'Glaube & Ethik':
            arrayGE.append(array[i])  
        if array[i][5] == 'Künste':
            arrayK.append(array[i])  
        if array[i][5] == 'Architektur & Garten':
            arrayAG.append(array[i])
        i += 1

    lengthArray = []
    lengthArray.append(len(arrayLU))
    lengthArray.append(len(arrayR))
    lengthArray.append(len(arrayKJ))
    lengthArray.append(len(arrayS))
    lengthArray.append(len(arrayGB))
    lengthArray.append(len(arrayGE))
    lengthArray.append(len(arrayK))
    lengthArray.append(len(arrayAG))
    maxNum = min(lengthArray)

    retArray = arrayLU[:maxNum] + arrayR[:maxNum] + arrayKJ[:maxNum] +arrayS[:maxNum] + arrayGB[:maxNum] + arrayGE[:maxNum] + arrayK[:maxNum] + arrayAG[:maxNum]
    return retArray

def featurize(text):
        j = 0
        k = 0
        nNouns = 0
        nVerbs = 0
        nAdjectives = 0
        nCommas = text.count(',')
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
        allHits = 0

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

                word = i[0].lower()
                if word in dictLU:
                        grLU += dictLU[word]
                        allHits += 1
                if word in dictR:
                        grR += dictR[word]
                        allHits += 1
                if word in dictKJ:
                        grKJ += dictKJ[word]
                        allHits += 1
                if word in dictS:
                        grS += dictS[word]
                        allHits += 1
                if word in dictGB:
                        grGB += dictGB[word]
                        allHits += 1
                if word in dictGE:
                        grGE += dictGE[word]
                        allHits += 1
                if word in dictK:
                        grK += dictK[word]
                        allHits += 1
                if word in dictAG:
                        grAG += dictAG[word]
                        allHits += 1

        if tockens == 0:
                tockens = 1

        rNouns = nNouns / tockens
        rVerbs = nVerbs / tockens
        rAdjectives = nAdjectives / tockens

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

        nSymE = text.count('€')
        nSymH = text.count('#')
        nSymD = text.count('$')
        nSymP = text.count('%')
        nSymPa = text.count('§')
        nSymA = text.count('&')
        nSymS = text.count('*')
        nSymQ = text.count('"')
        nSymDa = text.count('-')
        nSymDd = text.count(':')
        nSymSc = text.count(';')

        return j,k,rNouns,rVerbs,rAdjectives,nCommas,nSymE,nSymH,nSymD,nSymP,nSymPa,nSymA,nSymS,nSymQ,nSymDa,nSymDd,nSymSc,gdrLU,gdrR,gdrKJ,gdrS,gdrGB,gdrGE,gdrK,gdrAG

def createDataArray():
    global currPos
    global bookArray
    global tbookArray
    global data
    global tdata
    global isbnData
    global tisbnData
    global dataVec
    global tdataVec
    global testTexts
    global trainTexts
    
    currPos = 0
    # Creation of TrainDataFrame
    for _ in bookArray:
            wps,ns,rn,rv,ra,nc,nsyme,nsymH,nsymD,nsymp,nsympa,nsyma,nsyms,nsymQ,nsymda,nsymdd,nsymsc,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG = featurize(bookArray[currPos][0])
            trainTexts.append(bookArray[currPos][0])
            isbnData.append(bookArray[currPos][4])
            currPos += 1

    dataVector = vectorizer.transform(trainTexts)
    currPos = 0
    for _ in bookArray:
            vec = np.mean(dataVector[currPos].data)
            data.append([wps,ns,rn,rv,ra,nc,nsyme,nsymH,nsymD,nsymp,nsympa,nsyma,nsyms,nsymQ,nsymda,nsymdd,nsymsc,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG,bookArray[currPos][5]])
            currPos += 1

    # Creation of TestDataFrame
    currPos = 0
    for _ in tbookArray:
            wps,ns,rn,rv,ra,nc,nsyme,nsymH,nsymD,nsymp,nsympa,nsyma,nsyms,nsymQ,nsymda,nsymdd,nsymsc,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG = featurize(tbookArray[currPos][0])
            testTexts.append(tbookArray[currPos][0])
            tisbnData.append(tbookArray[currPos][4])
            currPos += 1

    tdataVector = vectorizer.transform(testTexts)
    currPos = 0
    for _ in tbookArray:
            vec = np.mean(tdataVector[currPos].data)
            tdata.append([wps,ns,rn,rv,ra,nc,nsyme,nsymH,nsymD,nsymp,nsympa,nsyma,nsyms,nsymQ,nsymda,nsymdd,nsymsc,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG,tbookArray[currPos][5]])
            currPos += 1

def createDataFrames():
    global data
    global tdata
    global isbnData
    global X_train
    global X_test
    global y_test
    global y_train

    # TrainDataFrame
    dataFrame=pd.DataFrame(data,columns=['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols€','NumberSymbolsH','NumberSymbolsD','NumberSymbols%','NumberSymbols§','NumberSymbols&','NumberSymbols*','NumberSymbolsQ','NumberSymbols-','NumberSymbols:','NumberSymbols;','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG','Genre'],dtype=float)

    # TestDataFrame
    tdataFrame=pd.DataFrame(tdata,columns=['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols€','NumberSymbolsH','NumberSymbolsD','NumberSymbols%','NumberSymbols§','NumberSymbols&','NumberSymbols*','NumberSymbolsQ','NumberSymbols-','NumberSymbols:','NumberSymbols;','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG','Genre'],dtype=float)

    # TrainData
    X_train=dataFrame[['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols€','NumberSymbolsH','NumberSymbolsD','NumberSymbols%','NumberSymbols§','NumberSymbols&','NumberSymbols*','NumberSymbolsQ','NumberSymbols-','NumberSymbols:','NumberSymbols;','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG']]
    y_train=dataFrame['Genre']

    # TestData
    X_test=tdataFrame[['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols€','NumberSymbolsH','NumberSymbolsD','NumberSymbols%','NumberSymbols§','NumberSymbols&','NumberSymbols*','NumberSymbolsQ','NumberSymbols-','NumberSymbols:','NumberSymbols;','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG']]
    y_test=tdataFrame['Genre']

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
    global y_predRF
    global estimator

    # RandomForestClassifier
    #randomForestClassifier=RandomForestClassifier(n_estimators=50,random_state=0,class_weight=[{0:1,"Literatur & Unterhaltung":1},{0:1,"Ratgeber":3},{0:1,"Kinderbuch & Jugendbuch":3},{0:1,"Sachbuch":3},{0:1,"Ganzheitliches Bewusstsein":3},{0:1,"Glaube & Ethik":3},{0:1,"Künste":3},{0:1,"Architektur & Garten":3}])
    #randomForestClassifier=RandomForestClassifier(n_estimators=50,random_state=0,class_weight=[{0:1,1:.3},{0:1,1:1},{0:1,1:1},{0:1,1:1},{0:1,1:1},{0:1,1:1},{0:1,1:1},{0:1,1:1}])
    #randomForestClassifier=RandomForestClassifier(n_estimators=50,random_state=0,class_weight=[{0:.3,1:1,2:1,3:1,4:1,5:1,6:1,7:1}])
    randomForestClassifier=RandomForestClassifier(n_estimators=50,random_state=0,verbose=10,n_jobs=2,bootstrap=True)
    randomForestClassifier.fit(X_train,y_train)

    y_predRF=randomForestClassifier.predict(X_test)

    print(randomForestClassifier.predict(np.asarray(X_test.loc[1,]).reshape(1,-1)))
    print(randomForestClassifier.predict_log_proba(np.asarray(X_test.loc[1,]).reshape(1,-1)))

    estimator = randomForestClassifier.estimators_[2]

    """
    gridSearchForest = RandomForestClassifier()
    params = {"n_estimators":[50,60,70],"max_depth": [5,8,15],"min_samples_leaf":[1,2,4]}
    clf = GridSearchCV(gridSearchForest,param_grid=params,cv=5)
    clf.fit(X_train,y_train)

    print(clf.best_params_)
    print(clf.best_score_)
    """

    if verbose:
        print("Feature Importance:",randomForestClassifier.feature_importances_)

        print("F-Score RandomForest:",metrics.f1_score(y_test, y_predRF,average='micro'))

        print("ConfusionMatrix RandomForest:\n",metrics.confusion_matrix(y_test, y_predRF,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"]))
        
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

def verboseOutput():
    global y_test
    global y_predRF
    prfs = metrics.precision_recall_fscore_support(y_test, y_predRF, average='micro')
    print(prfs[0],prfs[1],prfs[2],prfs[3])
    prfsLabel = metrics.precision_recall_fscore_support(y_test, y_predRF, average=None,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"])
    print("LU-precision:",prfsLabel[0][0])
    print("R-precision:",prfsLabel[0][1])
    print("KJ-precision:",prfsLabel[0][2])
    print("S-precision:",prfsLabel[0][3])
    print("GB-precision:",prfsLabel[0][4])
    print("GE-precision:",prfsLabel[0][5])
    print("K-precision:",prfsLabel[0][6])
    print("AG-precision:",prfsLabel[0][7])

    print("LU-recall:",prfsLabel[1][0])
    print("R-recall:",prfsLabel[1][1])
    print("KJ-recall:",prfsLabel[1][2])
    print("S-recall:",prfsLabel[1][3])
    print("GB-recall:",prfsLabel[1][4])
    print("GE-recall:",prfsLabel[1][5])
    print("K-recall:",prfsLabel[1][6])
    print("AG-recall:",prfsLabel[1][7])

    print("LU-fscore:",prfsLabel[2][0])
    print("R-fscore:",prfsLabel[2][1])
    print("KJ-fscore:",prfsLabel[2][2])
    print("S-fscore:",prfsLabel[2][3])
    print("GB-fscore:",prfsLabel[2][4])
    print("GE-fscore:",prfsLabel[2][5])
    print("K-fscore:",prfsLabel[2][6])
    print("AG-fscore:",prfsLabel[2][7])

    with open("verboseResults.txt","w") as file:
        file.write("Precision RandomForest:" + str(prfs[0]) + str("\n"))
        file.write("Recall RandomForest:" + str(prfs[1]) + str("\n"))
        file.write("F-Score RandomForest:" + str(prfs[2]) + str("\n"))
        file.write("ConfusionMatrix RandomForest:\n" + str(metrics.confusion_matrix(y_test, y_predRF,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"])) + str("\n"))

def generateFinalOutputFile():
    global tisbnData
    global y_predRF
    global estimator

    j = 0
    with open("finalOut.txt","w") as file:
        for i in tisbnData:
            file.write(i + str("\t") + str(y_predRF[j]) + str("\n"))
            j += 1
    export_graphviz(estimator, out_file='tree.dot', 
                feature_names = ['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols€','NumberSymbolsH','NumberSymbolsD','NumberSymbols%','NumberSymbols§','NumberSymbols&','NumberSymbols*','NumberSymbolsQ','NumberSymbols-','NumberSymbols:','NumberSymbols;','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG'],
                class_names = ["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"],
                rounded = True, proportion = False, 
                precision = 2, filled = True)

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
    start = timeit.default_timer()
    readDataOneFile()
    stopWordListRead()
    runNum = 0
    runRange = int(arg["num"])
    for runNum in range(runRange):
        print("Starting Run: " + str(runNum + 1))
        splitData()
        bookArray = permutateArray(bookArray)
        createTempDict()
        improveDict()
        print(dictAG)
        vectorizer = HashingVectorizer(stop_words=stopwords, alternate_sign=False,analyzer='word',ngram_range=(1,2),strip_accents='unicode',norm='l2')
        createDataArray()
        createDataFrames()
        trainClassifier()
        verboseOutput()
        generateFinalOutputFile()
    stop = timeit.default_timer()
    print("Runntime: ", stop - start)
if args.cv:
    readDataOneFile()
    stopWordListRead()
    splitData()
    createTempDict()
    createDataCrossVal()
    classifierCorssVal()
    verboseOutput()
