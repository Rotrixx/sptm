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

X_trainNoWPS = None
X_testNoWPS = None
y_testNoWPS = None
y_trainNoWPS = None

X_trainNoNS = None
X_testNoNS = None
y_testNoNS = None
y_trainNoNS = None

X_trainNoRPog = None
X_testNoRPog = None
y_testNoRPog = None
y_trainNoRPog = None

X_trainNoSym = None
X_testNoSym = None
y_testNoSym = None
y_trainNoSym = None

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
    # Creation of TrainDataFrame
    for _ in bookArray:
            #wps,ns,rn,rv,ra,nc,nsym,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG = featurize(bookArray[currPos][0])
            wps,ns,rn,rv,ra,nc,nsyme,nsymH,nsymD,nsymp,nsympa,nsyma,nsyms,nsymQ,nsymda,nsymdd,nsymsc,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG = featurize(bookArray[currPos][0])
            data.append([wps,ns,rn,rv,ra,nc,nsyme,nsymH,nsymD,nsymp,nsympa,nsyma,nsyms,nsymQ,nsymda,nsymdd,nsymsc,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG,bookArray[currPos][5]])
            isbnData.append(bookArray[currPos][4])
            currPos += 1
    # Creation of TestDataFrame
    currPos = 0
    for _ in tbookArray:
            # wps,ns,rn,rv,ra,nc,nsym,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG = featurize(tbookArray[currPos][0])
            wps,ns,rn,rv,ra,nc,nsyme,nsymH,nsymD,nsymp,nsympa,nsyma,nsyms,nsymQ,nsymda,nsymdd,nsymsc,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG = featurize(tbookArray[currPos][0])
            # tdata.append([wps,ns,rn,rv,ra,nc,nsym,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG,tbookArray[currPos][5]])
            tdata.append([wps,ns,rn,rv,ra,nc,nsyme,nsymH,nsymD,nsymp,nsympa,nsyma,nsyms,nsymQ,nsymda,nsymdd,nsymsc,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG,tbookArray[currPos][5]])
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

    global X_trainNoWPS
    global X_testNoWPS
    global y_testNoWPS
    global y_trainNoWPS

    global X_trainNoNS
    global X_testNoNS
    global y_testNoNS
    global y_trainNoNS

    global X_trainNoRPog
    global X_testNoRPog
    global y_testNoRPog
    global y_trainNoRPog

    global X_trainNoSym
    global X_testNoSym
    global y_testNoSym
    global y_trainNoSym

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

    #NOWPS
    # TrainData
    X_trainNoWPS=dataFrame[['NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols€','NumberSymbolsH','NumberSymbolsD','NumberSymbols%','NumberSymbols§','NumberSymbols&','NumberSymbols*','NumberSymbolsQ','NumberSymbols-','NumberSymbols:','NumberSymbols;','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG']]
    y_trainNoWPS=dataFrame['Genre']

    # TestData
    X_testNoWPS=tdataFrame[['NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols€','NumberSymbolsH','NumberSymbolsD','NumberSymbols%','NumberSymbols§','NumberSymbols&','NumberSymbols*','NumberSymbolsQ','NumberSymbols-','NumberSymbols:','NumberSymbols;','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG']]
    y_testNoWPS=tdataFrame['Genre']

    #NoNS
    # TrainData
    X_trainNoNS=dataFrame[['WordsPerSentence','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols€','NumberSymbolsH','NumberSymbolsD','NumberSymbols%','NumberSymbols§','NumberSymbols&','NumberSymbols*','NumberSymbolsQ','NumberSymbols-','NumberSymbols:','NumberSymbols;','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG']]
    y_trainNoNS=dataFrame['Genre']

    # TestData
    X_testNoNS=tdataFrame[['WordsPerSentence','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols€','NumberSymbolsH','NumberSymbolsD','NumberSymbols%','NumberSymbols§','NumberSymbols&','NumberSymbols*','NumberSymbolsQ','NumberSymbols-','NumberSymbols:','NumberSymbols;','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG']]
    y_testNoNS=tdataFrame['Genre']

    # NoRPOG
    # TrainData
    X_trainNoRPog=dataFrame[['WordsPerSentence','NumberSentences','NumberCommas','NumberSymbols€','NumberSymbolsH','NumberSymbolsD','NumberSymbols%','NumberSymbols§','NumberSymbols&','NumberSymbols*','NumberSymbolsQ','NumberSymbols-','NumberSymbols:','NumberSymbols;','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG']]
    y_trainNoRPog=dataFrame['Genre']

    # TestData
    X_testNoRPog=tdataFrame[['WordsPerSentence','NumberSentences','NumberCommas','NumberSymbols€','NumberSymbolsH','NumberSymbolsD','NumberSymbols%','NumberSymbols§','NumberSymbols&','NumberSymbols*','NumberSymbolsQ','NumberSymbols-','NumberSymbols:','NumberSymbols;','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG']]
    y_testNoRPog=tdataFrame['Genre']

    #NoSym
    # TrainData
    X_trainNoSym=dataFrame[['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG']]
    y_trainNoSym=dataFrame['Genre']

    # TestData
    X_testNoSym=tdataFrame[['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG']]
    y_testNoSym=tdataFrame['Genre']

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

    global X_trainNoWPS
    global X_testNoWPS
    global y_testNoWPS
    global y_trainNoWPS

    global X_trainNoNS
    global X_testNoNS
    global y_testNoNS
    global y_trainNoNS

    global X_trainNoRPog
    global X_testNoRPog
    global y_testNoRPog
    global y_trainNoRPog

    global X_trainNoSym
    global X_testNoSym
    global y_testNoSym
    global y_trainNoSym

    global y_predRF

    # RandomForestClassifier
    randomForestClassifier=RandomForestClassifier(n_estimators=50,oob_score=True,random_state=0,class_weight="balanced")
    randomForestClassifier.fit(X_train,y_train)

    randomForestClassifierNoWPS=RandomForestClassifier(n_estimators=50,oob_score=True,random_state=0,class_weight="balanced")
    randomForestClassifierNoWPS.fit(X_trainNoWPS,y_trainNoWPS)

    randomForestClassifierNoNS=RandomForestClassifier(n_estimators=50,oob_score=True,random_state=0,class_weight="balanced")
    randomForestClassifierNoNS.fit(X_trainNoNS,y_trainNoNS)

    randomForestClassifierNoRPog=RandomForestClassifier(n_estimators=50,oob_score=True,random_state=0,class_weight="balanced")
    randomForestClassifierNoRPog.fit(X_trainNoRPog,y_trainNoRPog)

    randomForestClassifierNoSym=RandomForestClassifier(n_estimators=50,oob_score=True,random_state=0,class_weight="balanced")
    randomForestClassifierNoSym.fit(X_trainNoSym,y_trainNoSym)

    y_predRF=randomForestClassifier.predict(X_test)
    y_predRFNoWPS=randomForestClassifierNoWPS.predict(X_testNoWPS)
    y_predRFNoNS=randomForestClassifierNoNS.predict(X_testNoNS)
    y_predRFNoRPog=randomForestClassifierNoRPog.predict(X_testNoRPog)
    y_predRFNoSym=randomForestClassifierNoSym.predict(X_testNoSym)

    if verbose:
        print("F-Score RandomForest:",metrics.f1_score(y_test, y_predRF,average='micro'))
        print("F-Score RandomForestNoWPS:",metrics.f1_score(y_testNoWPS, y_predRFNoWPS,average='micro'))
        print("F-Score RandomForestNoNS:",metrics.f1_score(y_testNoNS, y_predRFNoNS,average='micro'))
        print("F-Score RandomForestNoRPog:",metrics.f1_score(y_testNoRPog, y_predRFNoRPog,average='micro'))
        print("F-Score RandomForestNoSym:",metrics.f1_score(y_testNoSym, y_predRFNoSym,average='micro'))

        print("ConfusionMatrix RandomForest:\n",metrics.confusion_matrix(y_test, y_predRF,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"]))
        print("ConfusionMatrix RandomForestNoWPS:\n",metrics.confusion_matrix(y_testNoWPS, y_predRFNoWPS,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"]))
        print("ConfusionMatrix RandomForestNoNS:\n",metrics.confusion_matrix(y_testNoNS, y_predRFNoNS,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"]))
        print("ConfusionMatrix RandomForestNoRPog:\n",metrics.confusion_matrix(y_testNoRPog, y_predRFNoRPog,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"]))
        print("ConfusionMatrix RandomForestNoSym:\n",metrics.confusion_matrix(y_testNoSym, y_predRFNoSym,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"]))

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
    with open("verboseResults.txt","w") as file:
        file.write("F-Score RandomForest:" + str(metrics.f1_score(y_test, y_predRF,average='micro')) + str("\n"))
        file.write("ConfusionMatrix RandomForest:\n" + str(metrics.confusion_matrix(y_test, y_predRF,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"])) + str("\n"))
        
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
        verboseOutput()
if args.cv:
    readDataOneFile()
    stopWordListRead()
    splitData()
    createTempDict()
    createDataCrossVal()
    classifierCorssVal()
    verboseOutput()
