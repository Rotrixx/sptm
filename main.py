#!/usr/bin/python3.7

from textblob_de import TextBlobDE
import pandas as pd
import json
import argparse
import random
import timeit
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import metrics
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
multilabel = False
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
authorDict = {}
stopwords = None
estimator = None

#labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"]

def stopWordListRead():
    global stopwords
    """
    Stopwordliste wird aus Datei(Pfad: Data/stopwords_german.txt) eingelesen.
    """
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
    """
    Trainingsdaten werden aus Datei(Pfad: Data/blurbs_train.txt)eingelesen und in Array gespeichert.
    bookArray[pos][0] = Klappentext als String
    bookArray[pos][1] = Title als String 
    bookArray[pos][2] = Autor als String
    bookArray[pos][3] = Alle Kategorien in set()
    bookArray[pos][4] = ISBN als String
    bookArray[pos][5] = Erste Kategorie als String
    Leere Klappentexte und ISBN mit Startnummer 4 werden uebersprungen. 
    """
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
                            if multilabel:
                                for i in allCategoryStr:
                                    if bodyStr == '':
                                        continue
                                    if isbnStr.startswith('4'):
                                        continue
                                    bookArray.append((bodyStr + ' ' + titleStr,titleStr,authorStr,allCategoryStr,isbnStr,i))
                            else:
                                if bodyStr == '':
                                        continue
                                if isbnStr.startswith('4'):
                                        continue
                                bookArray.append((bodyStr + ' ' + titleStr,titleStr,authorStr,allCategoryStr,isbnStr,firstCategory))
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
                            firstComma = authorStr.find(',')
                            if firstComma != -1:
                                authorStr = authorStr[firstComma-1:]
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
    """
    Noch zu klassifizerende Testdaten werden aus Datei(Pfad: Data/blurbs_dev_participants.txt)eingelesen und in Array gespeichert.
    tbookArray[pos][0] = Klappentext als String
    tbookArray[pos][1] = Title als String 
    tbookArray[pos][2] = Autor als String
    tbookArray[pos][3] = Alle Kategorien in set()
    tbookArray[pos][4] = ISBN als String
    tbookArray[pos][5] = Erste Kategorie als String
    Leere Klappentexte und ISBN mit Startnummer 4 werden ?bersprungen.

    ToDo: Auf blurbs_dev_participants anpassen(Kategorien entfernen) 
    """
    with open('Data/blurbs_dev_participants.txt','r') as file:
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
                            if multilabel:
                                for i in tallCategoryStr:
                                    if tbodyStr == '':
                                        continue
                                    if tisbnStr.startswith('4'):
                                        continue
                                    tbookArray.append((tbodyStr + ' ' + ttitleStr,ttitleStr,tauthorStr,tallCategoryStr,tisbnStr,i))
                            else:
                                if tbodyStr == '':
                                        continue
                                if tisbnStr.startswith('4'):
                                        continue
                                tbookArray.append((tbodyStr + ' ' + ttitleStr,ttitleStr,tauthorStr,tallCategoryStr,tisbnStr,tfirstCategory))
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
                            firstComma = authorStr.find(',')
                            if firstComma != -1:
                                authorStr = authorStr[firstComma-1:]
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
                            tisbnStr += line
                            tisbnStr = tisbnStr[:-8]
                            tisbnStr = tisbnStr[6:]

def splitter(array, size):
    """
    Hilfsfunktion fuer splitData()
    """
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
    """
    Daten werden aufgeteilt in Trainings- und Testdaten.
    (10,000 Training)
    (Rest   Test)
    """
    random.shuffle(bookArray)
    helper = splitter(bookArray,10000)
    tbookArray = helper[1]
    bookArray = helper[0]
    print(len(tbookArray))
    print(len(bookArray))

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
        """
        Erstellung eines W?rterbucheintrages(Wort:Wert) f?r das mitgegebene Wort.
        W?rter die in der Stopwordliste auftreten werden gel?scht.
        Falls das Wort bereits im W?rterbuch steht, wird der Wert um eins erh?ht.
        Erstellung eines set() mit allen W?rtern.
        """

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
    """
    Gewichtung der Woerter in den Woerterbuechern
    ToDo Array mit allen Woerterbuechern erstellen und rueber-iterieren
    """

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
    global authorDict
    """
    Erstellung eines temporaeren Woerterbuches fuer Nomen,Verben und Adjektive.
    Alle Buchstaben werden kleingeschrieben. 

    Autorenwoerterbuch
    1 = Literatur & Unterhaltung
    2 = Ratgeber
    4 = Kinderbuch & Jugendbuch
    8 = Sachbuch
    16 = Ganzheitliches Bewusstsein
    32 = Glaube & Ethik
    64 = Künste
    128 = Architektur & Garten
    """
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

            if book[2] not in authorDict:
                authorDict[book[2]] = 0

            if book[5] == 'Literatur & Unterhaltung':
                authorDict[book[2]] += 1
            if book[5] == 'Ratgeber':
                authorDict[book[2]] += 2
            if book[5] == 'Kinderbuch & Jugendbuch':
                authorDict[book[2]] += 4
            if book[5] == 'Sachbuch':
                authorDict[book[2]] += 8
            if book[5] == 'Ganzheitliches Bewusstsein':
                authorDict[book[2]] += 16
            if book[5] == 'Glaube & Ethik':
                authorDict[book[2]] += 32
            if book[5] == 'Künste':
                authorDict[book[2]] += 64
            if book[5] == 'Architektur & Garten':
                authorDict[book[2]] += 128

def featurize(text,author):
        #global dictLU
        #global dictR
        #global dictKJ
        #global dictS
        #global dictGB
        #global dictGE
        #global dictK
        #global dictAG
        #global authorDict
        """
        Erstellung verschiedener Features:
        Woerter pro Satz
        Anzahl Saetze
        relative Haeufigkeiten von Nomen,Verben,Adjektiven
        Anzahl ausgewaehlten Symbolen(? # $ % ? & * " - : ; , )
        Genrewoerterbuchuebereinstimmungsraten
        """
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

        if author in authorDict:
                auth = authorDict[author]
        else:
                auth = 0

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

        return j,k,rNouns,rVerbs,rAdjectives,nCommas,nSymE,nSymH,nSymD,nSymP,nSymPa,nSymA,nSymS,nSymQ,nSymDa,nSymDd,nSymSc,gdrLU,gdrR,gdrKJ,gdrS,gdrGB,gdrGE,gdrK,gdrAG,auth

def createDataArray():
    global currPos
    global bookArray
    global tbookArray
    global data
    global tdata
    global isbnData
    global tisbnData
    """
    Erstellung eines Array mit den Featuren und Genre, zur Uebergabe an createDataFrame()
    """

    currPos = 0
    # Creation of TrainDataFrame
    for _ in bookArray:
            wps,ns,rn,rv,ra,nc,nsyme,nsymH,nsymD,nsymp,nsympa,nsyma,nsyms,nsymQ,nsymda,nsymdd,nsymsc,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG,auth = featurize(bookArray[currPos][0],bookArray[currPos][2])
            data.append([wps,ns,rn,rv,ra,nc,nsyme,nsymH,nsymD,nsymp,nsympa,nsyma,nsyms,nsymQ,nsymda,nsymdd,nsymsc,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG,auth,bookArray[currPos][5]])
            isbnData.append(bookArray[currPos][4])
            currPos += 1
    # Creation of TestDataFrame
    currPos = 0
    for _ in tbookArray:
            wps,ns,rn,rv,ra,nc,nsyme,nsymH,nsymD,nsymp,nsympa,nsyma,nsyms,nsymQ,nsymda,nsymdd,nsymsc,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG,auth = featurize(tbookArray[currPos][0],bookArray[currPos][2])
            tdata.append([wps,ns,rn,rv,ra,nc,nsyme,nsymH,nsymD,nsymp,nsympa,nsyma,nsyms,nsymQ,nsymda,nsymdd,nsymsc,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG,auth,tbookArray[currPos][5]])
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
    """
    Erstellung der DataFrames und aufteilen in X_train,X_test(Features) und y_train,y_test(Genres)
    """

    # TrainDataFrame
    dataFrame=pd.DataFrame(data,columns=['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols€','NumberSymbolsH','NumberSymbolsD','NumberSymbols%','NumberSymbols§','NumberSymbols&','NumberSymbols*','NumberSymbolsQ','NumberSymbols-','NumberSymbols:','NumberSymbols;','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG','Author','Genre'],dtype=float)

    # TestDataFrame
    tdataFrame=pd.DataFrame(tdata,columns=['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols€','NumberSymbolsH','NumberSymbolsD','NumberSymbols%','NumberSymbols§','NumberSymbols&','NumberSymbols*','NumberSymbolsQ','NumberSymbols-','NumberSymbols:','NumberSymbols;','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG','Author','Genre'],dtype=float)

    # TrainData
    X_train=dataFrame[['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols€','NumberSymbolsH','NumberSymbolsD','NumberSymbols%','NumberSymbols§','NumberSymbols&','NumberSymbols*','NumberSymbolsQ','NumberSymbols-','NumberSymbols:','NumberSymbols;','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG','Author']]
    y_train=dataFrame['Genre']

    # TestData
    X_test=tdataFrame[['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols€','NumberSymbolsH','NumberSymbolsD','NumberSymbols%','NumberSymbols§','NumberSymbols&','NumberSymbols*','NumberSymbolsQ','NumberSymbols-','NumberSymbols:','NumberSymbols;','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG','Author']]
    y_test=tdataFrame['Genre']

    print(dataFrame[:10])

def trainClassifier():
    global X_train
    global X_test
    global y_test
    global y_train
    global y_predRF
    global estimator
    """
    RandomForest Klassifikator trainieren und predicten.
    """

    randomForestClassifier=RandomForestClassifier(n_estimators=50,random_state=0,verbose=10,n_jobs=2)
    randomForestClassifier.fit(X_train,y_train)

    y_predRF=randomForestClassifier.predict(X_test)

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
        print(randomForestClassifier.feature_importances_)

        print("F-Score RandomForest:",metrics.f1_score(y_test, y_predRF,average='micro'))

        print("ConfusionMatrix RandomForest:\n",metrics.confusion_matrix(y_test, y_predRF,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"]))

def verboseOutput():
    global y_test
    global y_predRF
    """
    Zusaetzliche Valierungsausgabe in Terminal und Datei(OuputData/verboseOutput.txt)
    """
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
        file.write("LU-precision:" + str(prfsLabel[0][0]) + str("\n"))
        file.write("R-precision:" + str(prfsLabel[0][1]) + str("\n"))
        file.write("KJ-precision:" + str(prfsLabel[0][2]) + str("\n"))
        file.write("S-precision:" + str(prfsLabel[0][3]) + str("\n"))
        file.write("GB-precision:" + str(prfsLabel[0][4]) + str("\n"))
        file.write("GE-precision:" + str(prfsLabel[0][5]) + str("\n"))
        file.write("K-precision:" + str(prfsLabel[0][6]) + str("\n"))
        file.write("AG-precision:" + str(prfsLabel[0][7]) + str("\n"))

        file.write("LU-recall:" + str(prfsLabel[1][0]) + str("\n"))
        file.write("R-recall:" + str(prfsLabel[1][1]) + str("\n"))
        file.write("KJ-recall:" + str(prfsLabel[1][2]) + str("\n"))
        file.write("S-recall:" + str(prfsLabel[1][3]) + str("\n"))
        file.write("GB-recall:" + str(prfsLabel[1][4]) + str("\n"))
        file.write("GE-recall:" + str(prfsLabel[1][5]) + str("\n"))
        file.write("K-recall:" + str(prfsLabel[1][6]) + str("\n"))
        file.write("AG-recall:" + str(prfsLabel[1][7]) + str("\n"))

        file.write("LU-fscore:" + str(prfsLabel[2][0]) + str("\n"))
        file.write("R-fscore:" + str(prfsLabel[2][1]) + str("\n"))
        file.write("KJ-fscore:" + str(prfsLabel[2][2]) + str("\n"))
        file.write("S-fscore:" + str(prfsLabel[2][3]) + str("\n"))
        file.write("GB-fscore:" + str(prfsLabel[2][4]) + str("\n"))
        file.write("GE-fscore:" + str(prfsLabel[2][5]) + str("\n"))
        file.write("K-fscore:" + str(prfsLabel[2][6]) + str("\n"))
        file.write("AG-fscore:" + str(prfsLabel[2][7]) + str("\n"))
        file.write("Precision RandomForest:" + str(prfs[0]) + str("\n"))
        file.write("Recall RandomForest:" + str(prfs[1]) + str("\n"))
        file.write("F-Score RandomForest:" + str(prfs[2]) + str("\n"))
        file.write("ConfusionMatrix RandomForest:\n" + str(metrics.confusion_matrix(y_test, y_predRF,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"])) + str("\n"))

def generateFinalOutputFile():
    global tisbnData
    global y_test
    global X_test
    global y_predRF
    global estimator
    """
    Ausgabe Ergebnis-Datei und Baum 
    """

    j = 0
    with open("finalOut.txt","w") as file:
        for i in tisbnData:
            file.write(i + str("\t") + str(y_predRF[j]) + str("\n"))
            j += 1

    export_graphviz(estimator, out_file='tree.dot', 
                feature_names = ['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols€','NumberSymbolsH','NumberSymbolsD','NumberSymbols%','NumberSymbols§','NumberSymbols&','NumberSymbols*','NumberSymbolsQ','NumberSymbols-','NumberSymbols:','NumberSymbols;','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG','Author'],
                class_names = ["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"],
                rounded = True, proportion = False, 
                precision = 2, filled = True)

"""
Parser zur Ausfuehrung ueber das Terminal mit zusaetzlichen Angaben
"""
parser = argparse.ArgumentParser(description='sptm')
parser.add_argument("-v", help="activate verbose output", action="store_true")
parser.add_argument("-m", help="activate multilabel classification", action="store_true")
parser.add_argument("-x", help="n crossvalidation", action="store_true")
parser.add_argument("-val", help="validation", action="store_true")
args = parser.parse_args()

if args.v:
    verbose = True
if args.m:
    multilabel = True
if args.x:
    start = timeit.default_timer()
    readTrainData()
    stopWordListRead()
    splitData()
    createTempDict()
    improveDict()
    createDataArray()
    createDataFrames()
    trainClassifier()
    verboseOutput()
    generateFinalOutputFile()
    stop = timeit.default_timer()
    print("Runntime: ", stop - start)
if args.val:
    start = timeit.default_timer()
    readTrainData()
    readTestData()
    stopWordListRead()
    createTempDict()
    improveDict()
    createDataArray()
    createDataFrames()
    trainClassifier()
    verboseOutput()
    generateFinalOutputFile()
    stop = timeit.default_timer()
    print("Runntime: ", stop - start)