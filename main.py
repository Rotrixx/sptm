#!/usr/bin/python3.7

from textblob_de import TextBlobDE
import pandas as pd
import json
import argparse
import random
import timeit
import spacy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from multiprocessing import Pool

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
lemmatizeVerbs = False
lemmatizeNouns = False
lemmatizeAdjectives = False
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
stopwords = None
nlp = spacy.load('de_core_news_sm')

nounsList = ['NN','NNS','NNP','NNPS']
adjectiveList = ['JJ','JJR','JJS']
verbsList = ['VB','VBZ','VBP','VBD','VBN','VBG']

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
                        if not bodyStr:
                            continue
                        if isbnStr.startswith('4'):
                            continue
                        bookArray.append((bodyStr + ' ' + titleStr,titleStr,authorStr,allCategoryStr,isbnStr,i))
                else:
                    if not bodyStr:
                        continue
                    if isbnStr.startswith('4'):
                        continue
                    bookArray.append((bodyStr + ' ' + titleStr,titleStr,authorStr,allCategoryStr,isbnStr,firstCategory))
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
    global isbnData
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
                        tbookArray.append((tbodyStr + ' ' + ttitleStr,ttitleStr,tauthorStr,tallCategoryStr,tisbnStr,i))
                        isbnData.append(tisbnStr)
                else:
                    tbookArray.append((tbodyStr + ' ' + ttitleStr,ttitleStr,tauthorStr,tallCategoryStr,tisbnStr,tfirstCategory))
                    isbnData.append(tisbnStr)
            elif line.startswith('<body>'):
                tbodyStr += line
                tbodyStr = tbodyStr[:-8]
                tbodyStr = tbodyStr[6:]
            elif line.startswith('<title>'):
                ttitleStr += line
                ttitleStr = ttitleStr[:-9]
                ttitleStr = ttitleStr[7:]
            elif line.startswith('<authors>'):
                tauthorStr += line
                tauthorStr = tauthorStr[:-11]
                tauthorStr = tauthorStr[9:]
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
    global isbnData
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
    for i in tbookArray:
        isbnData.append(i[4])

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
    if bookArray[curr][5] == 'Literatur & Unterhaltung' and word not in stopwords:
        if word in dictLU:
            dictLU[word] += 1
        else:
            dictLU[word] = 1
    if bookArray[curr][5] == 'Ratgeber' and word not in stopwords:
        if word in dictR:
            dictR[word] += 1
        else:
            dictR[word] = 1
    if bookArray[curr][5] == 'Kinderbuch & Jugendbuch' and word not in stopwords:
        if word in dictKJ:
            dictKJ[word] += 1
        else:
            dictKJ[word] = 1
    if bookArray[curr][5] == 'Sachbuch' and word not in stopwords:
        if word in dictS:
            dictS[word] += 1
        else:
            dictS[word] = 1
    if bookArray[curr][5] == 'Ganzheitliches Bewusstsein' and word not in stopwords:
        if word in dictGB:
            dictGB[word] += 1
        else:
            dictGB[word] = 1
    if bookArray[curr][5] == 'Glaube & Ethik' and word not in stopwords:
        if word in dictGE:
            dictGE[word] += 1
        else:
            dictGE[word] = 1
    if bookArray[curr][5] == 'Künste' and word not in stopwords:
        if word in dictK:
            dictK[word] += 1
        else:
            dictK[word] = 1
    if bookArray[curr][5] == 'Architektur & Garten' and word not in stopwords:
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
            dictLU[word] = np.log(dictLU[word])/counter
        if word in dictR:
            dictR[word] = np.log(dictR[word])/counter
        if word in dictKJ:
            dictKJ[word] = np.log(dictKJ[word])/counter
        if word in dictS:
            dictS[word] = np.log(dictS[word])/counter
        if word in dictGB:
            dictGB[word] = np.log(dictGB[word])/counter
        if word in dictGE:
            dictGE[word] = np.log(dictGE[word])/counter
        if word in dictK:
            dictK[word] = np.log(dictK[word])/counter
        if word in dictAG:
            dictAG[word] = np.log(dictAG[word])/counter

def createTempDict():
    global bookArray
    global nounsList
    global verbsList
    global adjectiveList
    global curr
    global nlp
    """
    Erstellung eines temporaeren Woerterbuches fuer Nomen,Verben und Adjektive.
    Alle Buchstaben werden kleingeschrieben.
    """
    for book in bookArray:
        blob = TextBlobDE(book[0])
        textTockens = blob.tags
        for i in textTockens:
            if i[1] in nounsList:
                if lemmatizeNouns:
                    wordStr = str(i[0])
                    word = nlp(wordStr)
                    for token in word:
                        word = token.lemma_
                    addToDict(word.lower())
                else:
                    addToDict(i[0].lower())
            if i[1] in adjectiveList:
                if lemmatizeAdjectives:
                    wordStr = str(i[0])
                    word = nlp(wordStr)
                    for token in word:
                        word = token.lemma_
                    addToDict(word.lower())
                else:
                    addToDict(i[0].lower())
            if i[1] in verbsList:
                if lemmatizeVerbs:
                    wordStr = str(i[0])
                    word = nlp(wordStr)
                    for token in word:
                        word = token.lemma_
                    addToDict(word.lower())
                else:
                    addToDict(i[0].lower())
        curr += 1

def featurize(text):
    """
    Erstellung verschiedener Features:
    Woerter pro Satz
    Anzahl Saetze
    relative Haeufigkeiten von Nomen,Verben,Adjektiven
    Anzahl ausgewaehlten Symbolen(? # $ % ? & * " - : ; , )
    Genrewoerterbuchuebereinstimmungsraten
    """
    text = text[0]
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
        if i[1] in nounsList:
            if lemmatizeNouns:
                wordStr = str(i[0])
                word = nlp(wordStr)
                for token in word:
                    word = token.lemma_
            nNouns += 1
        elif i[1] in adjectiveList:
            if lemmatizeAdjectives:
                wordStr = str(i[0])
                word = nlp(wordStr)
                for token in word:
                    word = token.lemma_
            nAdjectives += 1
        elif i[1] in verbsList:
            if lemmatizeVerbs:
                wordStr = str(i[0])
                word = nlp(wordStr)
                for token in word:
                    word = token.lemma_
            nVerbs += 1

        if lemmatizeVerbs and i[1] in verbsList:
            word = word.lower()
        elif lemmatizeAdjectives and i[1] in adjectiveList:
            word = word.lower()
        elif lemmatizeNouns and i[1] in nounsList:
            word = word.lower()
        else:
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

    gdrLU = grLU# / allHits
    gdrR = grR# / allHits
    gdrKJ = grKJ# / allHits
    gdrS = grS# / allHits
    gdrGB = grGB #/ allHits
    gdrGE = grGE# / allHits
    gdrK = grK# / allHits
    gdrAG = grAG# / allHits

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

    return [j,k,rNouns,rVerbs,rAdjectives,nCommas,nSymE,nSymH,nSymD,nSymP,nSymPa,nSymA,nSymS,nSymQ,nSymDa,nSymDd,nSymSc,gdrLU,gdrR,gdrKJ,gdrS,gdrGB,gdrGE,gdrK,gdrAG]

def meanFeatureAll():
    global data
    meanLU = [0] * 25
    meanR = [0] * 25
    meanKJ = [0] * 25
    meanS = [0] * 25
    meanGB = [0] * 25
    meanGE = [0] * 25
    meanK = [0] * 25
    meanAG = [0] * 25

    counterLU = 0
    counterR = 0
    counterKJ = 0
    counterS = 0
    counterGB = 0
    counterGE = 0
    counterK = 0
    counterAG = 0

    for row in data:
        if row[25] == 'Literatur & Unterhaltung':
            for i in range(25):
                meanLU[i] += row[i]
            counterLU += 1
        if row[25] == 'Ratgeber':
            for i in range(25):
                meanR[i] += row[i]
            counterR += 1
        if row[25] == 'Kinderbuch & Jugendbuch':
            for i in range(25):
                meanKJ[i] += row[i]
            counterKJ += 1
        if row[25] == 'Sachbuch':
            for i in range(25):
                meanS[i] += row[i]
            counterS += 1
        if row[25] == 'Ganzheitliches Bewusstsein':
            for i in range(25):
                meanGB[i] += row[i]
            counterGB += 1
        if row[25] == 'Glaube & Ethik':
            for i in range(25):
                meanGE[i] += row[i]
            counterGE += 1
        if row[25] == 'Künste':
            for i in range(25):
                meanK[i] += row[i]
            counterK += 1
        if row[25] == 'Architektur & Garten':
            for i in range(25):
                meanAG[i] += row[i]
            counterAG += 1

    for i in range(25):
        meanLU[i] = (meanLU[i] / counterLU)
    for i in range(25):
        meanR[i] = (meanR[i] / counterR)
    for i in range(25):
        meanKJ[i] = (meanKJ[i] / counterKJ)
    for i in range(25):
        meanS[i] = (meanS[i] / counterS)
    for i in range(25):
        meanGB[i] = (meanGB[i] / counterGB)
    for i in range(25):
        meanGE[i] = (meanGE[i] / counterGE)
    for i in range(25):
        meanK[i] = (meanK[i] / counterK)
    for i in range(25):
        meanAG[i] = (meanAG[i] / counterAG)

    return meanLU,meanR,meanKJ,meanS,meanGB,meanGE,meanK,meanAG

def meanFeatures(row):
    meanDistLU = []
    meanDistR = []
    meanDistKJ = []
    meanDistS = []
    meanDistGB = []
    meanDistGE = []
    meanDistK = []
    meanDistAG = []
    helpp = []
    rowLen = len(row)-1
    for i in range(rowLen):
            row.append(abs(row[i] - meanLU[i]))
            meanDistLU.append(abs(row[i] - meanLU[i]))
    for i in range(rowLen):
            row.append(abs(row[i] - meanR[i]))
            meanDistR.append(abs(row[i] - meanLU[i]))
    for i in range(rowLen):
            row.append(abs(row[i] - meanKJ[i]))
            meanDistKJ.append(abs(row[i] - meanLU[i]))
    for i in range(rowLen):
            row.append(abs(row[i] - meanS[i]))
            meanDistS.append(abs(row[i] - meanLU[i]))
    for i in range(rowLen):
            row.append(abs(row[i] - meanGB[i]))
            meanDistGB.append(abs(row[i] - meanLU[i]))
    for i in range(rowLen):
            row.append(abs(row[i] - meanLU[i]))
            meanDistGE.append(abs(row[i] - meanGE[i]))
    for i in range(rowLen):
            row.append(abs(row[i] - meanK[i]))
            meanDistK.append(abs(row[i] - meanLU[i]))
    for i in range(rowLen):
            row.append(abs(row[i] - meanAG[i]))
            meanDistAG.append(abs(row[i] - meanLU[i]))

    for i in range(len(meanDistLU)):
        helpp.append(min(meanDistLU[i],meanDistR[i],meanDistKJ[i],meanDistS[i],meanDistGB[i],meanDistGE[i],meanDistK[i],meanDistAG[i]))

    for i in range(len(helpp)):
        if helpp[i] == meanDistLU[i]:
            row.append(0)
        elif helpp[i] == meanDistR[i]:
            row.append(1)
        elif helpp[i] == meanDistKJ[i]:
            row.append(2)
        elif helpp[i] == meanDistS[i]:
            row.append(3)
        elif helpp[i] == meanDistGB[i]:
            row.append(4)
        elif helpp[i] == meanDistGE[i]:
            row.append(5)
        elif helpp[i] == meanDistK[i]:
            row.append(6)
        elif helpp[i] == meanDistAG[i]:
            row.append(7)

def createTrainDataArray():
    global currPos
    global bookArray
    global isbnData
    global data
    """
    Erstellung eines Array mit den Featuren und Genre, zur Uebergabe an createDataFrame()
    """

    currPos = 0
    # Creation of TrainDataFrame
    with Pool(4) as pool:
        data = pool.map(featurize, bookArray)
    list(data)
    for book in bookArray:
        data[currPos].append(book[5])
        currPos += 1

def createTestDataArray():
    global currPos
    global tdata
    global tbookArray
    # Creation of TestDataFrame
    currPos = 0
    with Pool(4) as pool:
        tdata = pool.map(featurize, tbookArray)
    list(tdata)
    for book in tbookArray:
        tdata[currPos].append(book[5])
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
    dataFrame=pd.DataFrame(data,columns=['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols€','NumberSymbolsH','NumberSymbolsD','NumberSymbols%','NumberSymbols§','NumberSymbols&','NumberSymbols*','NumberSymbolsQ','NumberSymbols-','NumberSymbols:','NumberSymbols;','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG','Genre','LUdistToGenreWordsPerSentence','LUdistToGenreNumberSentences','LUdistToGenrePercentageNouns','LUdistToGenrePercentageVerbs','LUdistToGenrePercentageAdjectives','LUdistToGenreNumberCommas','LUdistToGenreNumberSymbols€','LUdistToGenreNumberSymbolsH','LUdistToGenreNumberSymbolsD','LUdistToGenreNumberSymbols%','LUdistToGenreNumberSymbols§','LUdistToGenreNumberSymbols&','LUdistToGenreNumberSymbols*','LUdistToGenreNumberSymbolsQ','LUdistToGenreNumberSymbols-','LUdistToGenreNumberSymbols:','LUdistToGenreNumberSymbols;','LUdistToGenreGenreRateLU','LUdistToGenreGenreRateR','LUdistToGenreGenreRateKJ','LUdistToGenreGenreRateS','LUdistToGenreGenreRateGB','LUdistToGenreGenreRateGE','LUdistToGenreGenreRateK','LUdistToGenreGenreRateAG','RdistToGenreWordsPerSentence','RdistToGenreNumberSentences','RdistToGenrePercentageNouns','RdistToGenrePercentageVerbs','RdistToGenrePercentageAdjectives','RdistToGenreNumberCommas','RdistToGenreNumberSymbols€','RdistToGenreNumberSymbolsH','RdistToGenreNumberSymbolsD','RdistToGenreNumberSymbols%','RdistToGenreNumberSymbols§','RdistToGenreNumberSymbols&','RdistToGenreNumberSymbols*','RdistToGenreNumberSymbolsQ','RdistToGenreNumberSymbols-','RdistToGenreNumberSymbols:','RdistToGenreNumberSymbols;','RdistToGenreGenreRateLU','RdistToGenreGenreRateR','RdistToGenreGenreRateKJ','RdistToGenreGenreRateS','RdistToGenreGenreRateGB','RdistToGenreGenreRateGE','RdistToGenreGenreRateK','RdistToGenreGenreRateAG','KJdistToGenreWordsPerSentence','KJdistToGenreNumberSentences','KJdistToGenrePercentageNouns','KJdistToGenrePercentageVerbs','KJdistToGenrePercentageAdjectives','KJdistToGenreNumberCommas','KJdistToGenreNumberSymbols€','KJdistToGenreNumberSymbolsH','KJdistToGenreNumberSymbolsD','KJdistToGenreNumberSymbols%','KJdistToGenreNumberSymbols§','KJdistToGenreNumberSymbols&','KJdistToGenreNumberSymbols*','KJdistToGenreNumberSymbolsQ','KJdistToGenreNumberSymbols-','KJdistToGenreNumberSymbols:','KJdistToGenreNumberSymbols;','KJdistToGenreGenreRateLU','KJdistToGenreGenreRateR','KJdistToGenreGenreRateKJ','KJdistToGenreGenreRateS','KJdistToGenreGenreRateGB','KJdistToGenreGenreRateGE','KJdistToGenreGenreRateK','KJdistToGenreGenreRateAG','SdistToGenreWordsPerSentence','SdistToGenreNumberSentences','SdistToGenrePercentageNouns','SdistToGenrePercentageVerbs','SdistToGenrePercentageAdjectives','SdistToGenreNumberCommas','SdistToGenreNumberSymbols€','SdistToGenreNumberSymbolsH','SdistToGenreNumberSymbolsD','SdistToGenreNumberSymbols%','SdistToGenreNumberSymbols§','SdistToGenreNumberSymbols&','SdistToGenreNumberSymbols*','SdistToGenreNumberSymbolsQ','SdistToGenreNumberSymbols-','SdistToGenreNumberSymbols:','SdistToGenreNumberSymbols;','SdistToGenreGenreRateLU','SdistToGenreGenreRateR','SdistToGenreGenreRateKJ','SdistToGenreGenreRateS','SdistToGenreGenreRateGB','SdistToGenreGenreRateGE','SdistToGenreGenreRateK','SdistToGenreGenreRateAG','GBdistToGenreWordsPerSentence','GBdistToGenreNumberSentences','GBdistToGenrePercentageNouns','GBdistToGenrePercentageVerbs','GBdistToGenrePercentageAdjectives','GBdistToGenreNumberCommas','GBdistToGenreNumberSymbols€','GBdistToGenreNumberSymbolsH','GBdistToGenreNumberSymbolsD','GBdistToGenreNumberSymbols%','GBdistToGenreNumberSymbols§','GBdistToGenreNumberSymbols&','GBdistToGenreNumberSymbols*','GBdistToGenreNumberSymbolsQ','GBdistToGenreNumberSymbols-','GBdistToGenreNumberSymbols:','GBdistToGenreNumberSymbols;','GBdistToGenreGenreRateLU','GBdistToGenreGenreRateR','GBdistToGenreGenreRateKJ','GBdistToGenreGenreRateS','GBdistToGenreGenreRateGB','GBdistToGenreGenreRateGE','GBdistToGenreGenreRateK','GBdistToGenreGenreRateAG','GEdistToGenreWordsPerSentence','GEdistToGenreNumberSentences','GEdistToGenrePercentageNouns','GEdistToGenrePercentageVerbs','GEdistToGenrePercentageAdjectives','GEdistToGenreNumberCommas','GEdistToGenreNumberSymbols€','GEdistToGenreNumberSymbolsH','GEdistToGenreNumberSymbolsD','GEdistToGenreNumberSymbols%','GEdistToGenreNumberSymbols§','GEdistToGenreNumberSymbols&','GEdistToGenreNumberSymbols*','GEdistToGenreNumberSymbolsQ','GEdistToGenreNumberSymbols-','GEdistToGenreNumberSymbols:','GEdistToGenreNumberSymbols;','GEdistToGenreGenreRateLU','GEdistToGenreGenreRateR','GEdistToGenreGenreRateKJ','GEdistToGenreGenreRateS','GEdistToGenreGenreRateGB','GEdistToGenreGenreRateGE','GEdistToGenreGenreRateK','GEdistToGenreGenreRateAG','KdistToGenreWordsPerSentence','KdistToGenreNumberSentences','KdistToGenrePercentageNouns','KdistToGenrePercentageVerbs','KdistToGenrePercentageAdjectives','KdistToGenreNumberCommas','KdistToGenreNumberSymbols€','KdistToGenreNumberSymbolsH','KdistToGenreNumberSymbolsD','KdistToGenreNumberSymbols%','KdistToGenreNumberSymbols§','KdistToGenreNumberSymbols&','KdistToGenreNumberSymbols*','KdistToGenreNumberSymbolsQ','KdistToGenreNumberSymbols-','KdistToGenreNumberSymbols:','KdistToGenreNumberSymbols;','KdistToGenreGenreRateLU','KdistToGenreGenreRateR','KdistToGenreGenreRateKJ','KdistToGenreGenreRateS','KdistToGenreGenreRateGB','KdistToGenreGenreRateGE','KdistToGenreGenreRateK','KdistToGenreGenreRateAG','AGdistToGenreWordsPerSentence','AGdistToGenreNumberSentences','AGdistToGenrePercentageNouns','AGdistToGenrePercentageVerbs','AGdistToGenrePercentageAdjectives','AGdistToGenreNumberCommas','AGdistToGenreNumberSymbols€','AGdistToGenreNumberSymbolsH','AGdistToGenreNumberSymbolsD','AGdistToGenreNumberSymbols%','AGdistToGenreNumberSymbols§','AGdistToGenreNumberSymbols&','AGdistToGenreNumberSymbols*','AGdistToGenreNumberSymbolsQ','AGdistToGenreNumberSymbols-','AGdistToGenreNumberSymbols:','AGdistToGenreNumberSymbols;','AGdistToGenreGenreRateLU','AGdistToGenreGenreRateR','AGdistToGenreGenreRateKJ','AGdistToGenreGenreRateS','AGdistToGenreGenreRateGB','AGdistToGenreGenreRateGE','AGdistToGenreGenreRateK','AGdistToGenreGenreRateAG','minDistToGenreWordsPerSentence','minDistToGenreNumberSentences','minDistToGenrePercentageNouns','minDistToGenrePercentageVerbs','minDistToGenrePercentageAdjectives','minDistToGenreNumberCommas','minDistToGenreNumberSymbols€','minDistToGenreNumberSymbolsH','minDistToGenreNumberSymbolsD','minDistToGenreNumberSymbols%','minDistToGenreNumberSymbols§','minDistToGenreNumberSymbols&','minDistToGenreNumberSymbols*','minDistToGenreNumberSymbolsQ','minDistToGenreNumberSymbols-','minDistToGenreNumberSymbols:','minDistToGenreNumberSymbols;','minDistToGenreGenreRateLU','minDistToGenreGenreRateR','minDistToGenreGenreRateKJ','minDistToGenreGenreRateS','minDistToGenreGenreRateGB','minDistToGenreGenreRateGE','minDistToGenreGenreRateK','minDistToGenreGenreRateAG'],dtype=float)

    # TestDataFrame
    tdataFrame=pd.DataFrame(tdata,columns=['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols€','NumberSymbolsH','NumberSymbolsD','NumberSymbols%','NumberSymbols§','NumberSymbols&','NumberSymbols*','NumberSymbolsQ','NumberSymbols-','NumberSymbols:','NumberSymbols;','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG','Genre','LUdistToGenreWordsPerSentence','LUdistToGenreNumberSentences','LUdistToGenrePercentageNouns','LUdistToGenrePercentageVerbs','LUdistToGenrePercentageAdjectives','LUdistToGenreNumberCommas','LUdistToGenreNumberSymbols€','LUdistToGenreNumberSymbolsH','LUdistToGenreNumberSymbolsD','LUdistToGenreNumberSymbols%','LUdistToGenreNumberSymbols§','LUdistToGenreNumberSymbols&','LUdistToGenreNumberSymbols*','LUdistToGenreNumberSymbolsQ','LUdistToGenreNumberSymbols-','LUdistToGenreNumberSymbols:','LUdistToGenreNumberSymbols;','LUdistToGenreGenreRateLU','LUdistToGenreGenreRateR','LUdistToGenreGenreRateKJ','LUdistToGenreGenreRateS','LUdistToGenreGenreRateGB','LUdistToGenreGenreRateGE','LUdistToGenreGenreRateK','LUdistToGenreGenreRateAG','RdistToGenreWordsPerSentence','RdistToGenreNumberSentences','RdistToGenrePercentageNouns','RdistToGenrePercentageVerbs','RdistToGenrePercentageAdjectives','RdistToGenreNumberCommas','RdistToGenreNumberSymbols€','RdistToGenreNumberSymbolsH','RdistToGenreNumberSymbolsD','RdistToGenreNumberSymbols%','RdistToGenreNumberSymbols§','RdistToGenreNumberSymbols&','RdistToGenreNumberSymbols*','RdistToGenreNumberSymbolsQ','RdistToGenreNumberSymbols-','RdistToGenreNumberSymbols:','RdistToGenreNumberSymbols;','RdistToGenreGenreRateLU','RdistToGenreGenreRateR','RdistToGenreGenreRateKJ','RdistToGenreGenreRateS','RdistToGenreGenreRateGB','RdistToGenreGenreRateGE','RdistToGenreGenreRateK','RdistToGenreGenreRateAG','KJdistToGenreWordsPerSentence','KJdistToGenreNumberSentences','KJdistToGenrePercentageNouns','KJdistToGenrePercentageVerbs','KJdistToGenrePercentageAdjectives','KJdistToGenreNumberCommas','KJdistToGenreNumberSymbols€','KJdistToGenreNumberSymbolsH','KJdistToGenreNumberSymbolsD','KJdistToGenreNumberSymbols%','KJdistToGenreNumberSymbols§','KJdistToGenreNumberSymbols&','KJdistToGenreNumberSymbols*','KJdistToGenreNumberSymbolsQ','KJdistToGenreNumberSymbols-','KJdistToGenreNumberSymbols:','KJdistToGenreNumberSymbols;','KJdistToGenreGenreRateLU','KJdistToGenreGenreRateR','KJdistToGenreGenreRateKJ','KJdistToGenreGenreRateS','KJdistToGenreGenreRateGB','KJdistToGenreGenreRateGE','KJdistToGenreGenreRateK','KJdistToGenreGenreRateAG','SdistToGenreWordsPerSentence','SdistToGenreNumberSentences','SdistToGenrePercentageNouns','SdistToGenrePercentageVerbs','SdistToGenrePercentageAdjectives','SdistToGenreNumberCommas','SdistToGenreNumberSymbols€','SdistToGenreNumberSymbolsH','SdistToGenreNumberSymbolsD','SdistToGenreNumberSymbols%','SdistToGenreNumberSymbols§','SdistToGenreNumberSymbols&','SdistToGenreNumberSymbols*','SdistToGenreNumberSymbolsQ','SdistToGenreNumberSymbols-','SdistToGenreNumberSymbols:','SdistToGenreNumberSymbols;','SdistToGenreGenreRateLU','SdistToGenreGenreRateR','SdistToGenreGenreRateKJ','SdistToGenreGenreRateS','SdistToGenreGenreRateGB','SdistToGenreGenreRateGE','SdistToGenreGenreRateK','SdistToGenreGenreRateAG','GBdistToGenreWordsPerSentence','GBdistToGenreNumberSentences','GBdistToGenrePercentageNouns','GBdistToGenrePercentageVerbs','GBdistToGenrePercentageAdjectives','GBdistToGenreNumberCommas','GBdistToGenreNumberSymbols€','GBdistToGenreNumberSymbolsH','GBdistToGenreNumberSymbolsD','GBdistToGenreNumberSymbols%','GBdistToGenreNumberSymbols§','GBdistToGenreNumberSymbols&','GBdistToGenreNumberSymbols*','GBdistToGenreNumberSymbolsQ','GBdistToGenreNumberSymbols-','GBdistToGenreNumberSymbols:','GBdistToGenreNumberSymbols;','GBdistToGenreGenreRateLU','GBdistToGenreGenreRateR','GBdistToGenreGenreRateKJ','GBdistToGenreGenreRateS','GBdistToGenreGenreRateGB','GBdistToGenreGenreRateGE','GBdistToGenreGenreRateK','GBdistToGenreGenreRateAG','GEdistToGenreWordsPerSentence','GEdistToGenreNumberSentences','GEdistToGenrePercentageNouns','GEdistToGenrePercentageVerbs','GEdistToGenrePercentageAdjectives','GEdistToGenreNumberCommas','GEdistToGenreNumberSymbols€','GEdistToGenreNumberSymbolsH','GEdistToGenreNumberSymbolsD','GEdistToGenreNumberSymbols%','GEdistToGenreNumberSymbols§','GEdistToGenreNumberSymbols&','GEdistToGenreNumberSymbols*','GEdistToGenreNumberSymbolsQ','GEdistToGenreNumberSymbols-','GEdistToGenreNumberSymbols:','GEdistToGenreNumberSymbols;','GEdistToGenreGenreRateLU','GEdistToGenreGenreRateR','GEdistToGenreGenreRateKJ','GEdistToGenreGenreRateS','GEdistToGenreGenreRateGB','GEdistToGenreGenreRateGE','GEdistToGenreGenreRateK','GEdistToGenreGenreRateAG','KdistToGenreWordsPerSentence','KdistToGenreNumberSentences','KdistToGenrePercentageNouns','KdistToGenrePercentageVerbs','KdistToGenrePercentageAdjectives','KdistToGenreNumberCommas','KdistToGenreNumberSymbols€','KdistToGenreNumberSymbolsH','KdistToGenreNumberSymbolsD','KdistToGenreNumberSymbols%','KdistToGenreNumberSymbols§','KdistToGenreNumberSymbols&','KdistToGenreNumberSymbols*','KdistToGenreNumberSymbolsQ','KdistToGenreNumberSymbols-','KdistToGenreNumberSymbols:','KdistToGenreNumberSymbols;','KdistToGenreGenreRateLU','KdistToGenreGenreRateR','KdistToGenreGenreRateKJ','KdistToGenreGenreRateS','KdistToGenreGenreRateGB','KdistToGenreGenreRateGE','KdistToGenreGenreRateK','KdistToGenreGenreRateAG','AGdistToGenreWordsPerSentence','AGdistToGenreNumberSentences','AGdistToGenrePercentageNouns','AGdistToGenrePercentageVerbs','AGdistToGenrePercentageAdjectives','AGdistToGenreNumberCommas','AGdistToGenreNumberSymbols€','AGdistToGenreNumberSymbolsH','AGdistToGenreNumberSymbolsD','AGdistToGenreNumberSymbols%','AGdistToGenreNumberSymbols§','AGdistToGenreNumberSymbols&','AGdistToGenreNumberSymbols*','AGdistToGenreNumberSymbolsQ','AGdistToGenreNumberSymbols-','AGdistToGenreNumberSymbols:','AGdistToGenreNumberSymbols;','AGdistToGenreGenreRateLU','AGdistToGenreGenreRateR','AGdistToGenreGenreRateKJ','AGdistToGenreGenreRateS','AGdistToGenreGenreRateGB','AGdistToGenreGenreRateGE','AGdistToGenreGenreRateK','AGdistToGenreGenreRateAG','minDistToGenreWordsPerSentence','minDistToGenreNumberSentences','minDistToGenrePercentageNouns','minDistToGenrePercentageVerbs','minDistToGenrePercentageAdjectives','minDistToGenreNumberCommas','minDistToGenreNumberSymbols€','minDistToGenreNumberSymbolsH','minDistToGenreNumberSymbolsD','minDistToGenreNumberSymbols%','minDistToGenreNumberSymbols§','minDistToGenreNumberSymbols&','minDistToGenreNumberSymbols*','minDistToGenreNumberSymbolsQ','minDistToGenreNumberSymbols-','minDistToGenreNumberSymbols:','minDistToGenreNumberSymbols;','minDistToGenreGenreRateLU','minDistToGenreGenreRateR','minDistToGenreGenreRateKJ','minDistToGenreGenreRateS','minDistToGenreGenreRateGB','minDistToGenreGenreRateGE','minDistToGenreGenreRateK','minDistToGenreGenreRateAG'],dtype=float)

    # TrainData
    X_train=dataFrame[['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols€','NumberSymbolsH','NumberSymbolsD','NumberSymbols%','NumberSymbols§','NumberSymbols&','NumberSymbols*','NumberSymbolsQ','NumberSymbols-','NumberSymbols:','NumberSymbols;','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG','LUdistToGenreWordsPerSentence','LUdistToGenreNumberSentences','LUdistToGenrePercentageNouns','LUdistToGenrePercentageVerbs','LUdistToGenrePercentageAdjectives','LUdistToGenreNumberCommas','LUdistToGenreNumberSymbols€','LUdistToGenreNumberSymbolsH','LUdistToGenreNumberSymbolsD','LUdistToGenreNumberSymbols%','LUdistToGenreNumberSymbols§','LUdistToGenreNumberSymbols&','LUdistToGenreNumberSymbols*','LUdistToGenreNumberSymbolsQ','LUdistToGenreNumberSymbols-','LUdistToGenreNumberSymbols:','LUdistToGenreNumberSymbols;','LUdistToGenreGenreRateLU','LUdistToGenreGenreRateR','LUdistToGenreGenreRateKJ','LUdistToGenreGenreRateS','LUdistToGenreGenreRateGB','LUdistToGenreGenreRateGE','LUdistToGenreGenreRateK','LUdistToGenreGenreRateAG','RdistToGenreWordsPerSentence','RdistToGenreNumberSentences','RdistToGenrePercentageNouns','RdistToGenrePercentageVerbs','RdistToGenrePercentageAdjectives','RdistToGenreNumberCommas','RdistToGenreNumberSymbols€','RdistToGenreNumberSymbolsH','RdistToGenreNumberSymbolsD','RdistToGenreNumberSymbols%','RdistToGenreNumberSymbols§','RdistToGenreNumberSymbols&','RdistToGenreNumberSymbols*','RdistToGenreNumberSymbolsQ','RdistToGenreNumberSymbols-','RdistToGenreNumberSymbols:','RdistToGenreNumberSymbols;','RdistToGenreGenreRateLU','RdistToGenreGenreRateR','RdistToGenreGenreRateKJ','RdistToGenreGenreRateS','RdistToGenreGenreRateGB','RdistToGenreGenreRateGE','RdistToGenreGenreRateK','RdistToGenreGenreRateAG','KJdistToGenreWordsPerSentence','KJdistToGenreNumberSentences','KJdistToGenrePercentageNouns','KJdistToGenrePercentageVerbs','KJdistToGenrePercentageAdjectives','KJdistToGenreNumberCommas','KJdistToGenreNumberSymbols€','KJdistToGenreNumberSymbolsH','KJdistToGenreNumberSymbolsD','KJdistToGenreNumberSymbols%','KJdistToGenreNumberSymbols§','KJdistToGenreNumberSymbols&','KJdistToGenreNumberSymbols*','KJdistToGenreNumberSymbolsQ','KJdistToGenreNumberSymbols-','KJdistToGenreNumberSymbols:','KJdistToGenreNumberSymbols;','KJdistToGenreGenreRateLU','KJdistToGenreGenreRateR','KJdistToGenreGenreRateKJ','KJdistToGenreGenreRateS','KJdistToGenreGenreRateGB','KJdistToGenreGenreRateGE','KJdistToGenreGenreRateK','KJdistToGenreGenreRateAG','SdistToGenreWordsPerSentence','SdistToGenreNumberSentences','SdistToGenrePercentageNouns','SdistToGenrePercentageVerbs','SdistToGenrePercentageAdjectives','SdistToGenreNumberCommas','SdistToGenreNumberSymbols€','SdistToGenreNumberSymbolsH','SdistToGenreNumberSymbolsD','SdistToGenreNumberSymbols%','SdistToGenreNumberSymbols§','SdistToGenreNumberSymbols&','SdistToGenreNumberSymbols*','SdistToGenreNumberSymbolsQ','SdistToGenreNumberSymbols-','SdistToGenreNumberSymbols:','SdistToGenreNumberSymbols;','SdistToGenreGenreRateLU','SdistToGenreGenreRateR','SdistToGenreGenreRateKJ','SdistToGenreGenreRateS','SdistToGenreGenreRateGB','SdistToGenreGenreRateGE','SdistToGenreGenreRateK','SdistToGenreGenreRateAG','GBdistToGenreWordsPerSentence','GBdistToGenreNumberSentences','GBdistToGenrePercentageNouns','GBdistToGenrePercentageVerbs','GBdistToGenrePercentageAdjectives','GBdistToGenreNumberCommas','GBdistToGenreNumberSymbols€','GBdistToGenreNumberSymbolsH','GBdistToGenreNumberSymbolsD','GBdistToGenreNumberSymbols%','GBdistToGenreNumberSymbols§','GBdistToGenreNumberSymbols&','GBdistToGenreNumberSymbols*','GBdistToGenreNumberSymbolsQ','GBdistToGenreNumberSymbols-','GBdistToGenreNumberSymbols:','GBdistToGenreNumberSymbols;','GBdistToGenreGenreRateLU','GBdistToGenreGenreRateR','GBdistToGenreGenreRateKJ','GBdistToGenreGenreRateS','GBdistToGenreGenreRateGB','GBdistToGenreGenreRateGE','GBdistToGenreGenreRateK','GBdistToGenreGenreRateAG','GEdistToGenreWordsPerSentence','GEdistToGenreNumberSentences','GEdistToGenrePercentageNouns','GEdistToGenrePercentageVerbs','GEdistToGenrePercentageAdjectives','GEdistToGenreNumberCommas','GEdistToGenreNumberSymbols€','GEdistToGenreNumberSymbolsH','GEdistToGenreNumberSymbolsD','GEdistToGenreNumberSymbols%','GEdistToGenreNumberSymbols§','GEdistToGenreNumberSymbols&','GEdistToGenreNumberSymbols*','GEdistToGenreNumberSymbolsQ','GEdistToGenreNumberSymbols-','GEdistToGenreNumberSymbols:','GEdistToGenreNumberSymbols;','GEdistToGenreGenreRateLU','GEdistToGenreGenreRateR','GEdistToGenreGenreRateKJ','GEdistToGenreGenreRateS','GEdistToGenreGenreRateGB','GEdistToGenreGenreRateGE','GEdistToGenreGenreRateK','GEdistToGenreGenreRateAG','KdistToGenreWordsPerSentence','KdistToGenreNumberSentences','KdistToGenrePercentageNouns','KdistToGenrePercentageVerbs','KdistToGenrePercentageAdjectives','KdistToGenreNumberCommas','KdistToGenreNumberSymbols€','KdistToGenreNumberSymbolsH','KdistToGenreNumberSymbolsD','KdistToGenreNumberSymbols%','KdistToGenreNumberSymbols§','KdistToGenreNumberSymbols&','KdistToGenreNumberSymbols*','KdistToGenreNumberSymbolsQ','KdistToGenreNumberSymbols-','KdistToGenreNumberSymbols:','KdistToGenreNumberSymbols;','KdistToGenreGenreRateLU','KdistToGenreGenreRateR','KdistToGenreGenreRateKJ','KdistToGenreGenreRateS','KdistToGenreGenreRateGB','KdistToGenreGenreRateGE','KdistToGenreGenreRateK','KdistToGenreGenreRateAG','AGdistToGenreWordsPerSentence','AGdistToGenreNumberSentences','AGdistToGenrePercentageNouns','AGdistToGenrePercentageVerbs','AGdistToGenrePercentageAdjectives','AGdistToGenreNumberCommas','AGdistToGenreNumberSymbols€','AGdistToGenreNumberSymbolsH','AGdistToGenreNumberSymbolsD','AGdistToGenreNumberSymbols%','AGdistToGenreNumberSymbols§','AGdistToGenreNumberSymbols&','AGdistToGenreNumberSymbols*','AGdistToGenreNumberSymbolsQ','AGdistToGenreNumberSymbols-','AGdistToGenreNumberSymbols:','AGdistToGenreNumberSymbols;','AGdistToGenreGenreRateLU','AGdistToGenreGenreRateR','AGdistToGenreGenreRateKJ','AGdistToGenreGenreRateS','AGdistToGenreGenreRateGB','AGdistToGenreGenreRateGE','AGdistToGenreGenreRateK','AGdistToGenreGenreRateAG','minDistToGenreWordsPerSentence','minDistToGenreNumberSentences','minDistToGenrePercentageNouns','minDistToGenrePercentageVerbs','minDistToGenrePercentageAdjectives','minDistToGenreNumberCommas','minDistToGenreNumberSymbols€','minDistToGenreNumberSymbolsH','minDistToGenreNumberSymbolsD','minDistToGenreNumberSymbols%','minDistToGenreNumberSymbols§','minDistToGenreNumberSymbols&','minDistToGenreNumberSymbols*','minDistToGenreNumberSymbolsQ','minDistToGenreNumberSymbols-','minDistToGenreNumberSymbols:','minDistToGenreNumberSymbols;','minDistToGenreGenreRateLU','minDistToGenreGenreRateR','minDistToGenreGenreRateKJ','minDistToGenreGenreRateS','minDistToGenreGenreRateGB','minDistToGenreGenreRateGE','minDistToGenreGenreRateK','minDistToGenreGenreRateAG']]
    y_train=dataFrame['Genre']

    # TestData
    X_test=tdataFrame[['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols€','NumberSymbolsH','NumberSymbolsD','NumberSymbols%','NumberSymbols§','NumberSymbols&','NumberSymbols*','NumberSymbolsQ','NumberSymbols-','NumberSymbols:','NumberSymbols;','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG','LUdistToGenreWordsPerSentence','LUdistToGenreNumberSentences','LUdistToGenrePercentageNouns','LUdistToGenrePercentageVerbs','LUdistToGenrePercentageAdjectives','LUdistToGenreNumberCommas','LUdistToGenreNumberSymbols€','LUdistToGenreNumberSymbolsH','LUdistToGenreNumberSymbolsD','LUdistToGenreNumberSymbols%','LUdistToGenreNumberSymbols§','LUdistToGenreNumberSymbols&','LUdistToGenreNumberSymbols*','LUdistToGenreNumberSymbolsQ','LUdistToGenreNumberSymbols-','LUdistToGenreNumberSymbols:','LUdistToGenreNumberSymbols;','LUdistToGenreGenreRateLU','LUdistToGenreGenreRateR','LUdistToGenreGenreRateKJ','LUdistToGenreGenreRateS','LUdistToGenreGenreRateGB','LUdistToGenreGenreRateGE','LUdistToGenreGenreRateK','LUdistToGenreGenreRateAG','RdistToGenreWordsPerSentence','RdistToGenreNumberSentences','RdistToGenrePercentageNouns','RdistToGenrePercentageVerbs','RdistToGenrePercentageAdjectives','RdistToGenreNumberCommas','RdistToGenreNumberSymbols€','RdistToGenreNumberSymbolsH','RdistToGenreNumberSymbolsD','RdistToGenreNumberSymbols%','RdistToGenreNumberSymbols§','RdistToGenreNumberSymbols&','RdistToGenreNumberSymbols*','RdistToGenreNumberSymbolsQ','RdistToGenreNumberSymbols-','RdistToGenreNumberSymbols:','RdistToGenreNumberSymbols;','RdistToGenreGenreRateLU','RdistToGenreGenreRateR','RdistToGenreGenreRateKJ','RdistToGenreGenreRateS','RdistToGenreGenreRateGB','RdistToGenreGenreRateGE','RdistToGenreGenreRateK','RdistToGenreGenreRateAG','KJdistToGenreWordsPerSentence','KJdistToGenreNumberSentences','KJdistToGenrePercentageNouns','KJdistToGenrePercentageVerbs','KJdistToGenrePercentageAdjectives','KJdistToGenreNumberCommas','KJdistToGenreNumberSymbols€','KJdistToGenreNumberSymbolsH','KJdistToGenreNumberSymbolsD','KJdistToGenreNumberSymbols%','KJdistToGenreNumberSymbols§','KJdistToGenreNumberSymbols&','KJdistToGenreNumberSymbols*','KJdistToGenreNumberSymbolsQ','KJdistToGenreNumberSymbols-','KJdistToGenreNumberSymbols:','KJdistToGenreNumberSymbols;','KJdistToGenreGenreRateLU','KJdistToGenreGenreRateR','KJdistToGenreGenreRateKJ','KJdistToGenreGenreRateS','KJdistToGenreGenreRateGB','KJdistToGenreGenreRateGE','KJdistToGenreGenreRateK','KJdistToGenreGenreRateAG','SdistToGenreWordsPerSentence','SdistToGenreNumberSentences','SdistToGenrePercentageNouns','SdistToGenrePercentageVerbs','SdistToGenrePercentageAdjectives','SdistToGenreNumberCommas','SdistToGenreNumberSymbols€','SdistToGenreNumberSymbolsH','SdistToGenreNumberSymbolsD','SdistToGenreNumberSymbols%','SdistToGenreNumberSymbols§','SdistToGenreNumberSymbols&','SdistToGenreNumberSymbols*','SdistToGenreNumberSymbolsQ','SdistToGenreNumberSymbols-','SdistToGenreNumberSymbols:','SdistToGenreNumberSymbols;','SdistToGenreGenreRateLU','SdistToGenreGenreRateR','SdistToGenreGenreRateKJ','SdistToGenreGenreRateS','SdistToGenreGenreRateGB','SdistToGenreGenreRateGE','SdistToGenreGenreRateK','SdistToGenreGenreRateAG','GBdistToGenreWordsPerSentence','GBdistToGenreNumberSentences','GBdistToGenrePercentageNouns','GBdistToGenrePercentageVerbs','GBdistToGenrePercentageAdjectives','GBdistToGenreNumberCommas','GBdistToGenreNumberSymbols€','GBdistToGenreNumberSymbolsH','GBdistToGenreNumberSymbolsD','GBdistToGenreNumberSymbols%','GBdistToGenreNumberSymbols§','GBdistToGenreNumberSymbols&','GBdistToGenreNumberSymbols*','GBdistToGenreNumberSymbolsQ','GBdistToGenreNumberSymbols-','GBdistToGenreNumberSymbols:','GBdistToGenreNumberSymbols;','GBdistToGenreGenreRateLU','GBdistToGenreGenreRateR','GBdistToGenreGenreRateKJ','GBdistToGenreGenreRateS','GBdistToGenreGenreRateGB','GBdistToGenreGenreRateGE','GBdistToGenreGenreRateK','GBdistToGenreGenreRateAG','GEdistToGenreWordsPerSentence','GEdistToGenreNumberSentences','GEdistToGenrePercentageNouns','GEdistToGenrePercentageVerbs','GEdistToGenrePercentageAdjectives','GEdistToGenreNumberCommas','GEdistToGenreNumberSymbols€','GEdistToGenreNumberSymbolsH','GEdistToGenreNumberSymbolsD','GEdistToGenreNumberSymbols%','GEdistToGenreNumberSymbols§','GEdistToGenreNumberSymbols&','GEdistToGenreNumberSymbols*','GEdistToGenreNumberSymbolsQ','GEdistToGenreNumberSymbols-','GEdistToGenreNumberSymbols:','GEdistToGenreNumberSymbols;','GEdistToGenreGenreRateLU','GEdistToGenreGenreRateR','GEdistToGenreGenreRateKJ','GEdistToGenreGenreRateS','GEdistToGenreGenreRateGB','GEdistToGenreGenreRateGE','GEdistToGenreGenreRateK','GEdistToGenreGenreRateAG','KdistToGenreWordsPerSentence','KdistToGenreNumberSentences','KdistToGenrePercentageNouns','KdistToGenrePercentageVerbs','KdistToGenrePercentageAdjectives','KdistToGenreNumberCommas','KdistToGenreNumberSymbols€','KdistToGenreNumberSymbolsH','KdistToGenreNumberSymbolsD','KdistToGenreNumberSymbols%','KdistToGenreNumberSymbols§','KdistToGenreNumberSymbols&','KdistToGenreNumberSymbols*','KdistToGenreNumberSymbolsQ','KdistToGenreNumberSymbols-','KdistToGenreNumberSymbols:','KdistToGenreNumberSymbols;','KdistToGenreGenreRateLU','KdistToGenreGenreRateR','KdistToGenreGenreRateKJ','KdistToGenreGenreRateS','KdistToGenreGenreRateGB','KdistToGenreGenreRateGE','KdistToGenreGenreRateK','KdistToGenreGenreRateAG','AGdistToGenreWordsPerSentence','AGdistToGenreNumberSentences','AGdistToGenrePercentageNouns','AGdistToGenrePercentageVerbs','AGdistToGenrePercentageAdjectives','AGdistToGenreNumberCommas','AGdistToGenreNumberSymbols€','AGdistToGenreNumberSymbolsH','AGdistToGenreNumberSymbolsD','AGdistToGenreNumberSymbols%','AGdistToGenreNumberSymbols§','AGdistToGenreNumberSymbols&','AGdistToGenreNumberSymbols*','AGdistToGenreNumberSymbolsQ','AGdistToGenreNumberSymbols-','AGdistToGenreNumberSymbols:','AGdistToGenreNumberSymbols;','AGdistToGenreGenreRateLU','AGdistToGenreGenreRateR','AGdistToGenreGenreRateKJ','AGdistToGenreGenreRateS','AGdistToGenreGenreRateGB','AGdistToGenreGenreRateGE','AGdistToGenreGenreRateK','AGdistToGenreGenreRateAG','minDistToGenreWordsPerSentence','minDistToGenreNumberSentences','minDistToGenrePercentageNouns','minDistToGenrePercentageVerbs','minDistToGenrePercentageAdjectives','minDistToGenreNumberCommas','minDistToGenreNumberSymbols€','minDistToGenreNumberSymbolsH','minDistToGenreNumberSymbolsD','minDistToGenreNumberSymbols%','minDistToGenreNumberSymbols§','minDistToGenreNumberSymbols&','minDistToGenreNumberSymbols*','minDistToGenreNumberSymbolsQ','minDistToGenreNumberSymbols-','minDistToGenreNumberSymbols:','minDistToGenreNumberSymbols;','minDistToGenreGenreRateLU','minDistToGenreGenreRateR','minDistToGenreGenreRateKJ','minDistToGenreGenreRateS','minDistToGenreGenreRateGB','minDistToGenreGenreRateGE','minDistToGenreGenreRateK','minDistToGenreGenreRateAG']]
    y_test=tdataFrame['Genre']

    with open("verboseTrainDataFrame.txt","w") as file:
        file.write(str(dataFrame.head(10)))

def trainClassifier():
    global X_train
    global X_test
    global y_test
    global y_train
    global y_predRF
    """
    RandomForest Klassifikator trainieren und predicten.
    """

    randomForestClassifier=RandomForestClassifier(n_estimators=100,max_depth=50,min_samples_leaf=1,bootstrap=False,criterion='gini',verbose=10,n_jobs=2)
    randomForestClassifier.fit(X_train,y_train)

    y_predRF=randomForestClassifier.predict(X_test)

    """
    gridSearchForest = RandomForestClassifier()
    params = {"n_estimators":[100],"max_depth": [20,30,40,50],"min_samples_leaf":[1,2],"bootstrap":[False]}
    clf = GridSearchCV(gridSearchForest,param_grid=params,cv=5)
    clf.fit(X_train,y_train)

    print(clf.best_params_)
    print(clf.best_score_)
    """

    print("F-Score RandomForest:",metrics.f1_score(y_test, y_predRF,average='micro'))

    if verbose:
        print(randomForestClassifier.feature_importances_)

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

    with open("OutputData/verboseResults.txt","w") as file:
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
    global isbnData
    global y_predRF
    """
    Ausgabe Ergebnis-Datei und Baum
    """

    j = 0
    with open("OutputData/finalOut.txt","w") as file:
        file.write("subtask: 1\n")
        for i in isbnData:
            file.write(i + str("\t") + str(y_predRF[j]) + str("\n"))
            j += 1

"""
Parser zur Ausfuehrung ueber das Terminal mit zusaetzlichen Angaben
"""
parser = argparse.ArgumentParser(description='sptm')
parser.add_argument("-lv", help="activate verb lemmatizing", action="store_true")
parser.add_argument("-ln", help="activate noun lemmatizing", action="store_true")
parser.add_argument("-la", help="activate adjective lemmatizing", action="store_true")
parser.add_argument("-v", help="activate verbose output", action="store_true")
parser.add_argument("-m", help="activate multilabel classification", action="store_true")
parser.add_argument("-x", help="simulation of real use", action="store_true")
parser.add_argument("-x10", help="10 cross val with all train data", action="store_true")
parser.add_argument("-val", help="validation", action="store_true")
args = parser.parse_args()
if args.lv:
    lemmatizeVerbs = True
if args.ln:
    lemmatizeNouns = True
if args.la:
    lemmatizeAdjectives = True
if args.v:
    verbose = True
if args.m:
    multilabel = True
if args.x10:
    start = timeit.default_timer()
    readTrainData()
    stopWordListRead()
    print("Done reading files.", timeit.default_timer() - start)
    createTempDict()
    improveDict()
    print("Done creating dictionary.", timeit.default_timer() - start)
    createTrainDataArray()
    meanLU,meanR,meanKJ,meanS,meanGB,meanGE,meanK,meanAG = meanFeatureAll()
    for row in data:
        meanFeatures(row)
    createDataFrames()
    print("Done creating DataFrame.", timeit.default_timer() - start)
    randomForestClassifier=RandomForestClassifier(n_estimators=100,max_depth=50,min_samples_leaf=1,bootstrap=False,criterion='gini',n_jobs=2)
    crossVal = cross_val_score(randomForestClassifier,X_train,y_train,cv=10,scoring='f1_micro')
    print(crossVal)
    print("10-Cross:",np.mean(crossVal))
    stop = timeit.default_timer()
    print("Runntime: ", stop - start)
if args.x:
    start = timeit.default_timer()
    readTrainData()
    stopWordListRead()
    splitData()
    print(isbnData)
    print("Done reading files.", timeit.default_timer() - start)
    createTempDict()
    improveDict()
    print("Done creating dictionary.", timeit.default_timer() - start)
    createTrainDataArray()
    createTestDataArray()
    meanLU,meanR,meanKJ,meanS,meanGB,meanGE,meanK,meanAG = meanFeatureAll()
    for row in data:
        meanFeatures(row)
    for row in tdata:
        meanFeatures(row)
    createDataFrames()
    print("Done creating DataFrame.", timeit.default_timer() - start)
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
    print("Done reading files.", timeit.default_timer() - start)
    createTempDict()
    improveDict()
    print("Done creating dictionary.", timeit.default_timer() - start)
    createTrainDataArray()
    createTestDataArray()
    meanLU,meanR,meanKJ,meanS,meanGB,meanGE,meanK,meanAG = meanFeatureAll()
    for row in data:
        meanFeatures(row)
    for row in tdata:
        meanFeatures(row)
    createDataFrames()
    print("Done creating DataFrame.", timeit.default_timer() - start)
    trainClassifier()
    generateFinalOutputFile()
    stop = timeit.default_timer()
    print("Runntime: ", stop - start)
