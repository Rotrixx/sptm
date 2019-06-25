import json
import argparse
import random
import timeit
from multiprocessing import Pool
import spacy
import pandas as pd
import numpy as np
from textblob_de import TextBlobDE
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

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
tbookArray = []
tdata = []
tisbnData = []

allWords = set()
allBiWords = set()
allTriWords = set()
allQuadroWords = set()
allPentaWords = set()
allAuthors = set()

# Other
useSigmoid = False
useGomp = False
useOldRec = False
lemmatizeVerbs = False
lemmatizeNouns = False
lemmatizeAdjectives = False
verbose = False
multilabel = False
shuffle = False
multiprocessing = False
multiOut = False
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
biDictLU = {}
biDictR = {}
biDictKJ = {}
biDictS = {}
biDictGB = {}
biDictGE = {}
biDictK = {}
biDictAG = {}
triDictLU = {}
triDictR = {}
triDictKJ = {}
triDictS = {}
triDictGB = {}
triDictGE = {}
triDictK = {}
triDictAG = {}
quadroDictLU = {}
quadroDictR = {}
quadroDictKJ = {}
quadroDictS = {}
quadroDictGB = {}
quadroDictGE = {}
quadroDictK = {}
quadroDictAG = {}
pentaDictLU = {}
pentaDictR = {}
pentaDictKJ = {}
pentaDictS = {}
pentaDictGB = {}
pentaDictGE = {}
pentaDictK = {}
pentaDictAG = {}
auDictLU = {}
auDictR = {}
auDictKJ = {}
auDictS = {}
auDictGB = {}
auDictGE = {}
auDictK = {}
auDictAG = {}
authorRatesTrain = []
authorRatesTest = []
biGenreRatesTrain = []
biGenreRatesTest = []
triGenreRatesTrain = []
triGenreRatesTest = []
quadroGenreRatesTrain = []
quadroGenreRatesTest = []
pentaGenreRatesTrain = []
pentaGenreRatesTest = []
weights = {}
stopwords = None
nlp = spacy.load('de_core_news_sm')

nounsList = ['NN', 'NNS', 'NNP', 'NNPS']
adjectiveList = ['JJ', 'JJR', 'JJS']
verbsList = ['VB', 'VBZ', 'VBP', 'VBD', 'VBN', 'VBG']

phraseList = ['NP','PP','VP','ADVP','ADJP','PNP','SBAR','PRT']

#labels=["Literatur & Unterhaltung", "Ratgeber", "Kinderbuch & Jugendbuch", "Sachbuch", "Ganzheitliches Bewusstsein", "Glaube & Ethik", "Künste", "Architektur & Garten"]

def stopWordListRead():
    global stopwords
    """
    Stopwordliste wird aus Datei(Pfad: Data/stopwords_german.txt) eingelesen.
    """
    with open('Data/stopwords_german.txt', 'r') as file:
        stopwords = json.load(file)

def readData(dataFile):
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
    array = []
    with open(dataFile, 'r') as file:
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
                #"""
                if not bodyStr:
                    continue
                if isbnStr.startswith('4'):
                    continue
                #"""
                array.append((bodyStr + ' ' + titleStr, titleStr, authorStr, allCategoryStr, isbnStr, firstCategory))
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
    return array

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
    (10, 000 Training)
    (Rest   Test)
    """
    if shuffle:
        random.shuffle(bookArray)
    helper = splitter(bookArray, 14488)
    tbookArray = helper[1]
    bookArray = helper[0]
    print(len(tbookArray))
    print(len(bookArray))
    for i in tbookArray:
        isbnData.append(i[4])

    
    if multilabel:
        array = []
        for book in bookArray:
            for i in book[3]:
                array.append((book[0], book[1], book[2], 0, book[4], i))
        bookArray = array

    j = 0
    with open("../evaluation/input/gold.txt", "w") as file:
        file.write("subtask_a\n")
        for i in isbnData:
            labelStr = ''
            labels = tbookArray[j][3]
            for k in labels:
                labelStr +=  str("\t") + str(k)
            file.write(i + str(labelStr) + str("\n"))
            j += 1

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
    Falls das Wort bereits im W?rterbuch steht,  wird der Wert um eins erh?ht.
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

def addToBiDict(word):
    global curr
    global biDictLU
    global biDictR
    global biDictKJ
    global biDictS
    global biDictGB
    global biDictGE
    global biDictK
    global biDictAG
    global allBiWords
    """
    Erstellung eines W?rterbucheintrages(Wort:Wert) f?r das mitgegebene Wort.
    W?rter die in der Stopwordliste auftreten werden gel?scht.
    Falls das Wort bereits im W?rterbuch steht,  wird der Wert um eins erh?ht.
    Erstellung eines set() mit allen W?rtern.
    """
    if bookArray[curr][5] == 'Literatur & Unterhaltung':
        if word in biDictLU:
            biDictLU[word] += 1
        else:
            biDictLU[word] = 1
    if bookArray[curr][5] == 'Ratgeber':
        if word in biDictR:
            biDictR[word] += 1
        else:
            biDictR[word] = 1
    if bookArray[curr][5] == 'Kinderbuch & Jugendbuch':
        if word in biDictKJ:
            biDictKJ[word] += 1
        else:
            biDictKJ[word] = 1
    if bookArray[curr][5] == 'Sachbuch':
        if word in biDictS:
            biDictS[word] += 1
        else:
            biDictS[word] = 1
    if bookArray[curr][5] == 'Ganzheitliches Bewusstsein':
        if word in biDictGB:
            biDictGB[word] += 1
        else:
            biDictGB[word] = 1
    if bookArray[curr][5] == 'Glaube & Ethik':
        if word in biDictGE:
            biDictGE[word] += 1
        else:
            biDictGE[word] = 1
    if bookArray[curr][5] == 'Künste':
        if word in biDictK:
            biDictK[word] += 1
        else:
            biDictK[word] = 1
    if bookArray[curr][5] == 'Architektur & Garten':
        if word in biDictAG:
            biDictAG[word] += 1
        else:
            biDictAG[word] = 1
    allBiWords.add(word)

def addToTriDict(word):
    global curr
    global triDictLU
    global triDictR
    global triDictKJ
    global triDictS
    global triDictGB
    global triDictGE
    global triDictK
    global triDictAG
    global allTriWords
    """
    Erstellung eines W?rterbucheintrages(Wort:Wert) f?r das mitgegebene Wort.
    W?rter die in der Stopwordliste auftreten werden gel?scht.
    Falls das Wort bereits im W?rterbuch steht,  wird der Wert um eins erh?ht.
    Erstellung eines set() mit allen W?rtern.
    """
    if bookArray[curr][5] == 'Literatur & Unterhaltung':
        if word in triDictLU:
            triDictLU[word] += 1
        else:
            triDictLU[word] = 1
    if bookArray[curr][5] == 'Ratgeber':
        if word in triDictR:
            triDictR[word] += 1
        else:
            triDictR[word] = 1
    if bookArray[curr][5] == 'Kinderbuch & Jugendbuch':
        if word in triDictKJ:
            triDictKJ[word] += 1
        else:
            triDictKJ[word] = 1
    if bookArray[curr][5] == 'Sachbuch':
        if word in triDictS:
            triDictS[word] += 1
        else:
            triDictS[word] = 1
    if bookArray[curr][5] == 'Ganzheitliches Bewusstsein':
        if word in triDictGB:
            triDictGB[word] += 1
        else:
            triDictGB[word] = 1
    if bookArray[curr][5] == 'Glaube & Ethik':
        if word in triDictGE:
            triDictGE[word] += 1
        else:
            triDictGE[word] = 1
    if bookArray[curr][5] == 'Künste':
        if word in triDictK:
            triDictK[word] += 1
        else:
            triDictK[word] = 1
    if bookArray[curr][5] == 'Architektur & Garten':
        if word in triDictAG:
            triDictAG[word] += 1
        else:
            triDictAG[word] = 1
    allTriWords.add(word)

def addToQuadroDict(word):
    global curr
    global quadroDictLU
    global quadroDictR
    global quadroDictKJ
    global quadroDictS
    global quadroDictGB
    global quadroDictGE
    global quadroDictK
    global quadroDictAG
    global allQuadroWords
    """
    Erstellung eines W?rterbucheintrages(Wort:Wert) f?r das mitgegebene Wort.
    W?rter die in der Stopwordliste auftreten werden gel?scht.
    Falls das Wort bereits im W?rterbuch steht,  wird der Wert um eins erh?ht.
    Erstellung eines set() mit allen W?rtern.
    """
    if bookArray[curr][5] == 'Literatur & Unterhaltung':
        if word in quadroDictLU:
            quadroDictLU[word] += 1
        else:
            quadroDictLU[word] = 1
    if bookArray[curr][5] == 'Ratgeber':
        if word in quadroDictR:
            quadroDictR[word] += 1
        else:
            quadroDictR[word] = 1
    if bookArray[curr][5] == 'Kinderbuch & Jugendbuch':
        if word in quadroDictKJ:
            quadroDictKJ[word] += 1
        else:
            quadroDictKJ[word] = 1
    if bookArray[curr][5] == 'Sachbuch':
        if word in quadroDictS:
            quadroDictS[word] += 1
        else:
            quadroDictS[word] = 1
    if bookArray[curr][5] == 'Ganzheitliches Bewusstsein':
        if word in quadroDictGB:
            quadroDictGB[word] += 1
        else:
            quadroDictGB[word] = 1
    if bookArray[curr][5] == 'Glaube & Ethik':
        if word in quadroDictGE:
            quadroDictGE[word] += 1
        else:
            quadroDictGE[word] = 1
    if bookArray[curr][5] == 'Künste':
        if word in quadroDictK:
            quadroDictK[word] += 1
        else:
            quadroDictK[word] = 1
    if bookArray[curr][5] == 'Architektur & Garten':
        if word in quadroDictAG:
            quadroDictAG[word] += 1
        else:
            quadroDictAG[word] = 1
    allQuadroWords.add(word)

def addToPentaDict(word):
    global curr
    global pentaDictLU
    global pentaDictR
    global pentaDictKJ
    global pentaDictS
    global pentaDictGB
    global pentaDictGE
    global pentaDictK
    global pentaDictAG
    global allPentaWords
    """
    Erstellung eines W?rterbucheintrages(Wort:Wert) f?r das mitgegebene Wort.
    W?rter die in der Stopwordliste auftreten werden gel?scht.
    Falls das Wort bereits im W?rterbuch steht,  wird der Wert um eins erh?ht.
    Erstellung eines set() mit allen W?rtern.
    """
    if bookArray[curr][5] == 'Literatur & Unterhaltung':
        if word in pentaDictLU:
            pentaDictLU[word] += 1
        else:
            pentaDictLU[word] = 1
    if bookArray[curr][5] == 'Ratgeber':
        if word in pentaDictR:
            pentaDictR[word] += 1
        else:
            pentaDictR[word] = 1
    if bookArray[curr][5] == 'Kinderbuch & Jugendbuch':
        if word in pentaDictKJ:
            pentaDictKJ[word] += 1
        else:
            pentaDictKJ[word] = 1
    if bookArray[curr][5] == 'Sachbuch':
        if word in pentaDictS:
            pentaDictS[word] += 1
        else:
            pentaDictS[word] = 1
    if bookArray[curr][5] == 'Ganzheitliches Bewusstsein':
        if word in pentaDictGB:
            pentaDictGB[word] += 1
        else:
            pentaDictGB[word] = 1
    if bookArray[curr][5] == 'Glaube & Ethik':
        if word in pentaDictGE:
            pentaDictGE[word] += 1
        else:
            pentaDictGE[word] = 1
    if bookArray[curr][5] == 'Künste':
        if word in pentaDictK:
            pentaDictK[word] += 1
        else:
            pentaDictK[word] = 1
    if bookArray[curr][5] == 'Architektur & Garten':
        if word in pentaDictAG:
            pentaDictAG[word] += 1
        else:
            pentaDictAG[word] = 1
    allPentaWords.add(word)

def sig(x):
    return np.exp(x) / np.exp(x)+1

def gomp(x):
    a = np.exp(1)
    b = 4
    c = 0.736
    return (a*np.exp(1)**(-b*(np.exp(1))**(-(c*x))))

def improvedSig(w, c):
    return np.log(1.5*w) / sig(c)

def improvedGomp(w, c):
    return np.log(1.5*w) / gomp(c)

def improveDict():
    global dictLU
    global dictR
    global dictKJ
    global dictS
    global dictGB
    global dictGE
    global dictK
    global dictAG

    global biDictLU
    global biDictR
    global biDictKJ
    global biDictS
    global biDictGB
    global biDictGE
    global biDictK
    global biDictAG

    global triDictLU
    global triDictR
    global triDictKJ
    global triDictS
    global triDictGB
    global triDictGE
    global triDictK
    global triDictAG

    global quadroDictLU
    global quadroDictR
    global quadroDictKJ
    global quadroDictS
    global quadroDictGB
    global quadroDictGE
    global quadroDictK
    global quadroDictAG

    global pentaDictLU
    global pentaDictR
    global pentaDictKJ
    global pentaDictS
    global pentaDictGB
    global pentaDictGE
    global pentaDictK
    global pentaDictAG

    global auDictLU
    global auDictR
    global auDictKJ
    global auDictS
    global auDictGB
    global auDictGE
    global auDictK
    global auDictAG

    global allWords
    global allBiWords
    global allTriWords
    global allQuadroWords
    global allPentaWords
    global allAuthors

    cutoff = 0.56
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

        if useSigmoid:
            if word in dictLU:
                dictLU[word] = improvedSig(dictLU[word], counter)
            if word in dictR:
                dictR[word] =improvedSig(dictR[word], counter)
            if word in dictKJ:
                dictKJ[word] = improvedSig(dictKJ[word], counter)
            if word in dictS:
                dictS[word] =improvedSig(dictS[word], counter)
            if word in dictGB:
                dictGB[word] = improvedSig(dictGB[word], counter)
            if word in dictGE:
                dictGE[word] = improvedSig(dictGE[word], counter)
            if word in dictK:
                dictK[word] =improvedSig(dictK[word], counter)
            if word in dictAG:
                dictAG[word] = improvedSig(dictAG[word], counter)
        elif useGomp:
            if word in dictLU:
                dictLU[word] = improvedGomp(dictLU[word], counter)
            if word in dictR:
                dictR[word] =improvedGomp(dictR[word], counter)
            if word in dictKJ:
                dictKJ[word] = improvedGomp(dictKJ[word], counter)
            if word in dictS:
                dictS[word] =improvedGomp(dictS[word], counter)
            if word in dictGB:
                dictGB[word] = improvedGomp(dictGB[word], counter)
            if word in dictGE:
                dictGE[word] = improvedGomp(dictGE[word], counter)
            if word in dictK:
                dictK[word] =improvedGomp(dictK[word], counter)
            if word in dictAG:
                dictAG[word] = improvedGomp(dictAG[word], counter)
        elif useOldRec:
            if word in dictLU:
                dictLU[word] = (np.log(1.5*dictLU[word])/(counter))
            if word in dictR:
                dictR[word] = (np.log(1.5*dictR[word])/(counter))
            if word in dictKJ:
                dictKJ[word] = (np.log(1.5*dictKJ[word])/(counter))
            if word in dictS:
                dictS[word] = (np.log(1.5*dictS[word])/(counter))
            if word in dictGB:
                dictGB[word] = (np.log(1.5*dictGB[word])/(counter))
            if word in dictGE:
                dictGE[word] = (np.log(1.5*dictGE[word])/(counter))
            if word in dictK:
                dictK[word] = (np.log(1.5*dictK[word])/(counter))
            if word in dictAG:
                dictAG[word] = (np.log(1.5*dictAG[word])/(counter))
        else:
            if word in dictLU:
                dictLU[word] = (np.log(1.5*dictLU[word])/(sig(counter)))
            if word in dictR:
                dictR[word] = (np.log(1.5*dictR[word])/(np.exp(np.log10(counter*4))))
            if word in dictKJ:
                dictKJ[word] = (np.log(1.5*dictKJ[word])/(np.exp(np.log10(counter*4))))
            if word in dictS:
                dictS[word] = (np.log(1.5*dictS[word])/(np.exp(np.log10(counter*4))))
            if word in dictGB:
                dictGB[word] = (np.log(1.5*dictGB[word])/(np.exp(np.log10(counter*4))))
            if word in dictGE:
                dictGE[word] = (np.log(1.5*dictGE[word])/(np.exp(np.log10(counter*4))))
            if word in dictK:
                dictK[word] = (np.log(1.5*dictK[word])/(np.exp(np.log10(counter*4))))
            if word in dictAG:
                dictAG[word] = (np.log(1.5*dictAG[word])/(np.exp(np.log10(counter*4))))

        if word in dictLU and dictLU[word] <= cutoff:
            dictLU[word] = 0
        if word in dictR and dictR[word] <= cutoff:
            dictR[word] = 0
        if word in dictKJ and dictKJ[word] <= cutoff:
            dictKJ[word] = 0
        if word in dictS and dictS[word] <= cutoff:
            dictS[word] = 0
        if word in dictGB and dictGB[word] <= cutoff:
            dictGB[word] = 0
        if word in dictGE and dictGE[word] <= cutoff:
            dictGE[word] = 0
        if word in dictK and dictK[word] <= cutoff:
            dictK[word] = 0
        if word in dictAG and dictAG[word] <= cutoff:
            dictAG[word] = 0


    """
    for word in allTriWords:
        counter = 0
        if word in triDictLU:
            counter += 1
        if word in triDictR:
            counter += 1
        if word in triDictKJ:
            counter += 1
        if word in triDictS:
            counter += 1
        if word in triDictGB:
            counter += 1
        if word in triDictGE:
            counter += 1
        if word in triDictK:
            counter += 1
        if word in triDictAG:
            counter += 1

        if word in triDictLU:
            triDictLU[word] = improvedGomp(triDictLU[word],(counter))
        if word in triDictR:
            triDictR[word] = improvedGomp(triDictR[word],(counter))
        if word in triDictKJ:
            triDictKJ[word] = improvedGomp(triDictKJ[word],(counter))
        if word in triDictS:
            triDictS[word] = improvedGomp(triDictS[word],(counter))
        if word in triDictGB:
            triDictGB[word] = improvedGomp(triDictGB[word],(counter))
        if word in triDictGE:
            triDictGE[word] = improvedGomp(triDictGE[word],(counter))
        if word in triDictK:
            triDictK[word] = improvedGomp(triDictK[word],(counter))
        if word in triDictAG:
            triDictAG[word] = improvedGomp(triDictAG[word],(counter))

        if word in triDictLU and triDictLU[word] <= cutoff:
            triDictLU[word] = 0
        if word in triDictR and triDictR[word] <= cutoff:
            triDictR[word] = 0
        if word in triDictKJ and triDictKJ[word] <= cutoff:
            triDictKJ[word] = 0
        if word in triDictS and triDictS[word] <= cutoff:
            triDictS[word] = 0
        if word in triDictGB and triDictGB[word] <= cutoff:
            triDictGB[word] = 0
        if word in triDictGE and triDictGE[word] <= cutoff:
            triDictGE[word] = 0
        if word in triDictK and triDictK[word] <= cutoff:
            triDictK[word] = 0
        if word in triDictAG and triDictAG[word] <= cutoff:
            triDictAG[word] = 0

    for word in allQuadroWords:
        counter = 0
        if word in quadroDictLU:
            counter += 1
        if word in quadroDictR:
            counter += 1
        if word in quadroDictKJ:
            counter += 1
        if word in quadroDictS:
            counter += 1
        if word in quadroDictGB:
            counter += 1
        if word in quadroDictGE:
            counter += 1
        if word in quadroDictK:
            counter += 1
        if word in quadroDictAG:
            counter += 1

        if word in quadroDictLU:
            quadroDictLU[word] = improvedGomp(quadroDictLU[word],(counter))
        if word in quadroDictR:
            quadroDictR[word] = improvedGomp(quadroDictR[word],(counter))
        if word in quadroDictKJ:
            quadroDictKJ[word] = improvedGomp(quadroDictKJ[word],(counter))
        if word in quadroDictS:
            quadroDictS[word] = improvedGomp(quadroDictS[word],(counter))
        if word in quadroDictGB:
            quadroDictGB[word] = improvedGomp(quadroDictGB[word],(counter))
        if word in quadroDictGE:
            quadroDictGE[word] = improvedGomp(quadroDictGE[word],(counter))
        if word in quadroDictK:
            quadroDictK[word] = improvedGomp(quadroDictK[word],(counter))
        if word in quadroDictAG:
            quadroDictAG[word] = improvedGomp(quadroDictAG[word],(counter))

        if word in quadroDictLU and quadroDictLU[word] <= cutoff:
            quadroDictLU[word] = 0
        if word in quadroDictR and quadroDictR[word] <= cutoff:
            quadroDictR[word] = 0
        if word in quadroDictKJ and quadroDictKJ[word] <= cutoff:
            quadroDictKJ[word] = 0
        if word in quadroDictS and quadroDictS[word] <= cutoff:
            quadroDictS[word] = 0
        if word in quadroDictGB and quadroDictGB[word] <= cutoff:
            quadroDictGB[word] = 0
        if word in quadroDictGE and quadroDictGE[word] <= cutoff:
            quadroDictGE[word] = 0
        if word in quadroDictK and quadroDictK[word] <= cutoff:
            quadroDictK[word] = 0
        if word in quadroDictAG and quadroDictAG[word] <= cutoff:
            quadroDictAG[word] = 0

    for word in allPentaWords:
        counter = 0
        if word in pentaDictLU:
            counter += 1
        if word in pentaDictR:
            counter += 1
        if word in pentaDictKJ:
            counter += 1
        if word in pentaDictS:
            counter += 1
        if word in pentaDictGB:
            counter += 1
        if word in pentaDictGE:
            counter += 1
        if word in pentaDictK:
            counter += 1
        if word in pentaDictAG:
            counter += 1

        if word in pentaDictLU:
            pentaDictLU[word] = improvedGomp(pentaDictLU[word],(counter))
        if word in pentaDictR:
            pentaDictR[word] = improvedGomp(pentaDictR[word],(counter))
        if word in pentaDictKJ:
            pentaDictKJ[word] = improvedGomp(pentaDictKJ[word],(counter))
        if word in pentaDictS:
            pentaDictS[word] = improvedGomp(pentaDictS[word],(counter))
        if word in pentaDictGB:
            pentaDictGB[word] = improvedGomp(pentaDictGB[word],(counter))
        if word in pentaDictGE:
            pentaDictGE[word] = improvedGomp(pentaDictGE[word],(counter))
        if word in pentaDictK:
            pentaDictK[word] = improvedGomp(pentaDictK[word],(counter))
        if word in pentaDictAG:
            pentaDictAG[word] = improvedGomp(pentaDictAG[word],(counter))

        if word in pentaDictLU and pentaDictLU[word] <= cutoff:
            pentaDictLU[word] = 0
        if word in pentaDictR and pentaDictR[word] <= cutoff:
            pentaDictR[word] = 0
        if word in pentaDictKJ and pentaDictKJ[word] <= cutoff:
            pentaDictKJ[word] = 0
        if word in pentaDictS and pentaDictS[word] <= cutoff:
            pentaDictS[word] = 0
        if word in pentaDictGB and pentaDictGB[word] <= cutoff:
            pentaDictGB[word] = 0
        if word in pentaDictGE and pentaDictGE[word] <= cutoff:
            pentaDictGE[word] = 0
        if word in pentaDictK and pentaDictK[word] <= cutoff:
            pentaDictK[word] = 0
        if word in pentaDictAG and pentaDictAG[word] <= cutoff:
            pentaDictAG[word] = 0

    for word in allBiWords:
        counter = 0
        if word in biDictLU:
            counter += 1
        if word in biDictR:
            counter += 1
        if word in biDictKJ:
            counter += 1
        if word in biDictS:
            counter += 1
        if word in biDictGB:
            counter += 1
        if word in biDictGE:
            counter += 1
        if word in biDictK:
            counter += 1
        if word in biDictAG:
            counter += 1

        if word in biDictLU:
            biDictLU[word] = improvedGomp(biDictLU[word],(counter))
        if word in biDictR:
            biDictR[word] = improvedGomp(biDictR[word],(counter))
        if word in biDictKJ:
            biDictKJ[word] = improvedGomp(biDictKJ[word],(counter))
        if word in biDictS:
            biDictS[word] = improvedGomp(biDictS[word],(counter))
        if word in biDictGB:
            biDictGB[word] = improvedGomp(biDictGB[word],(counter))
        if word in biDictGE:
            biDictGE[word] = improvedGomp(biDictGE[word],(counter))
        if word in biDictK:
            biDictK[word] = improvedGomp(biDictK[word],(counter))
        if word in biDictAG:
            biDictAG[word] = improvedGomp(biDictAG[word],(counter))

        if word in biDictLU and biDictLU[word] <= cutoff:
            biDictLU[word] = 0
        if word in biDictR and biDictR[word] <= cutoff:
            biDictR[word] = 0
        if word in biDictKJ and biDictKJ[word] <= cutoff:
            biDictKJ[word] = 0
        if word in biDictS and biDictS[word] <= cutoff:
            biDictS[word] = 0
        if word in biDictGB and biDictGB[word] <= cutoff:
            biDictGB[word] = 0
        if word in biDictGE and biDictGE[word] <= cutoff:
            biDictGE[word] = 0
        if word in biDictK and biDictK[word] <= cutoff:
            biDictK[word] = 0
        if word in biDictAG and biDictAG[word] <= cutoff:
            biDictAG[word] = 0
    """

    for authorD in allAuthors:
        counter = 0
        if authorD in auDictLU:
            counter += 1
        if authorD in auDictR:
            counter += 1
        if authorD in auDictKJ:
            counter += 1
        if authorD in auDictS:
            counter += 1
        if authorD in auDictGB:
            counter += 1
        if authorD in auDictGE:
            counter += 1
        if authorD in auDictK:
            counter += 1
        if authorD in auDictAG:
            counter += 1

        if authorD in auDictLU:
            auDictLU[authorD] = improvedGomp(auDictLU[authorD],(counter))
        if authorD in auDictR:
            auDictR[authorD] = improvedGomp(auDictR[authorD],(counter))
        if authorD in auDictKJ:
            auDictKJ[authorD] = improvedGomp(auDictKJ[authorD],(counter))
        if authorD in auDictS:
            auDictS[authorD] = improvedGomp(auDictS[authorD],(counter))
        if authorD in auDictGB:
            auDictGB[authorD] = improvedGomp(auDictGB[authorD],(counter))
        if authorD in auDictGE:
            auDictGE[authorD] = improvedGomp(auDictGE[authorD],(counter))
        if authorD in auDictK:
            auDictK[authorD] = improvedGomp(auDictK[authorD],(counter))
        if authorD in auDictAG:
            auDictAG[authorD] = improvedGomp(auDictAG[authorD],(counter))

def addToAuthorDict(author):
    global curr
    global auDictLU
    global auDictR
    global auDictKJ
    global auDictS
    global auDictGB
    global auDictGE
    global auDictK
    global auDictAG
    global allAuthors

    if bookArray[curr][5] == 'Literatur & Unterhaltung':
        if author in auDictLU:
            auDictLU[author] += 1
        else:
            auDictLU[author] = 1
    if bookArray[curr][5] == 'Ratgeber':
        if author in auDictR:
            auDictR[author] += 1
        else:
            auDictR[author] = 1
    if bookArray[curr][5] == 'Kinderbuch & Jugendbuch':
        if author in auDictKJ:
            auDictKJ[author] += 1
        else:
            auDictKJ[author] = 1
    if bookArray[curr][5] == 'Sachbuch':
        if author in auDictS:
            auDictS[author] += 1
        else:
            auDictS[author] = 1
    if bookArray[curr][5] == 'Ganzheitliches Bewusstsein':
        if author in auDictGB:
            auDictGB[author] += 1
        else:
            auDictGB[author] = 1
    if bookArray[curr][5] == 'Glaube & Ethik':
        if author in auDictGE:
            auDictGE[author] += 1
        else:
            auDictGE[author] = 1
    if bookArray[curr][5] == 'Künste':
        if author in auDictK:
            auDictK[author] += 1
        else:
            auDictK[author] = 1
    if bookArray[curr][5] == 'Architektur & Garten':
        if author in auDictAG:
            auDictAG[author] += 1
        else:
            auDictAG[author] = 1
    allAuthors.add(author)

def createTempDict():
    global bookArray
    global nounsList
    global verbsList
    global adjectiveList
    global phraseList
    global curr
    global nlp
    """
    Erstellung eines temporaeren Woerterbuches fuer Nomen, Verben und Adjektive.
    Alle Buchstaben werden kleingeschrieben.
    """
    for book in bookArray:
        blob = TextBlobDE(book[0])
        textTockens = blob.tags
        for i in textTockens:
            word = i[0].lower()
            if i[1] in nounsList:
                if lemmatizeNouns:
                    wordStr = str(i[0])
                    word = nlp(wordStr)
                    for token in word:
                        word = token.lemma_
                    addToDict(word.lower())
                else:
                    addToDict(word)

            if i[1] in adjectiveList:
                if lemmatizeAdjectives:
                    wordStr = str(i[0])
                    word = nlp(wordStr)
                    for token in word:
                        word = token.lemma_
                    addToDict(word.lower())
                else:
                    addToDict(word)

            if i[1] in verbsList:
                if lemmatizeVerbs:
                    wordStr = str(i[0])
                    word = nlp(wordStr)
                    for token in word:
                        word = token.lemma_
                    addToDict(word.lower())
                else:
                    addToDict(word)

            try:
                pos = textTockens.index(i)
                biGram1 = str(textTockens[pos - 1][0] + ' ' +  i[0])
                biGram2 = str(i[0] + ' ' +  textTockens[pos + 1][0])
                triGram1 = str(textTockens[pos - 1][0] + ' ' +  i[0] + ' ' + textTockens[pos + 1][0])
                triGram2 = str(i[0] + ' ' + textTockens[pos + 1][0] + ' ' + textTockens[pos + 2][0])
                triGram3 = str(textTockens[pos - 2][0] + ' ' + textTockens[pos - 1][0] + ' ' + i[0])
                quadroGram1 = str(textTockens[pos - 1][0] + ' ' +  i[0] + ' ' + textTockens[pos + 1][0] + ' ' + textTockens[pos + 2][0])
                quadroGram2 = str(i[0] + ' ' + textTockens[pos + 1][0] + ' ' + textTockens[pos + 2][0] + ' ' + textTockens[pos + 3][0])
                quadroGram3 = str(textTockens[pos - 3][0] + ' ' + textTockens[pos - 2][0] + ' ' + textTockens[pos - 1][0] + ' ' + i[0])
                quadroGram4 = str(textTockens[pos - 2][0] + ' ' + textTockens[pos - 1][0] + ' ' + i[0] + ' ' + textTockens[pos + 1][0])
                pentaGram1 = str(textTockens[pos - 1][0] + ' ' +  i[0] + ' ' + textTockens[pos + 1][0] + ' ' + textTockens[pos + 2][0] + ' ' + textTockens[pos + 3][0])
                pentaGram2 = str(i[0] + ' ' + textTockens[pos + 1][0] + ' ' + textTockens[pos + 2][0] + ' ' + textTockens[pos + 3][0] + ' ' + textTockens[pos + 4][0])
                pentaGram3 = str(textTockens[pos - 4][0] + ' ' + textTockens[pos - 3][0] + ' ' + textTockens[pos - 2][0] + ' ' + textTockens[pos - 1][0] + ' ' + i[0])
                pentaGram4 = str(textTockens[pos - 2][0] + ' ' + textTockens[pos - 1][0] + ' ' + i[0] + ' ' + textTockens[pos + 1][0] + ' ' + textTockens[pos + 2][0])
                pentaGram4 = str(textTockens[pos - 3][0] + ' ' + textTockens[pos - 2][0] + ' ' + textTockens[pos - 1][0] + ' ' + i[0] + ' ' + textTockens[pos + 1][0])
            except:
                pass                    
            addToDict(biGram1)
            addToDict(biGram2)
            addToDict(triGram1)
            addToDict(triGram2)
            addToDict(triGram3)
            addToDict(quadroGram1)
            addToDict(quadroGram2)
            addToDict(quadroGram3)
            addToDict(quadroGram4)
            addToDict(pentaGram1)
            addToDict(pentaGram2)
            addToDict(pentaGram3)
            addToDict(pentaGram4)
            addToDict(pentaGram4)

        authorsList = str(book[2])
        authorsListStrs = authorsList.split(",")
        for author in authorsListStrs:
            if author == "":
                continue
            addToAuthorDict(author)

        curr += 1

def featurize(text):
    """
    Erstellung verschiedener Features:
    Woerter pro Satz
    Anzahl Saetze
    relative Haeufigkeiten von Nomen, Verben, Adjektiven
    Anzahl ausgewaehlten Symbolen(? # $ % ? & * " - : ; ,  )
    Genrewoerterbuchuebereinstimmungsraten
    """
    text = text[0]
    authors = text[2].split(",")
    j = 0
    j2 = 0
    k = 0
    nNouns = 0
    nVerbs = 0
    nAdjectives = 0
    nCommas = text.count(', ')
    tockens = 0
    rCharCount = 0
    maxCharCount = 0
    blob = TextBlobDE(text)
    for sentence in blob.sentences:
        k += 1
        for word in sentence.words:
            wordl = len(word)
            if wordl > maxCharCount:
                maxCharCount = wordl
            rCharCount += wordl
            j += 1
            j2 += 1

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
    allBiHits = 0
    allTriHits = 0
    allQuadroHits = 0
    allPentaHits = 0
    allAuHits = 0

    agrLU = 0
    agrR = 0
    agrKJ = 0
    agrS = 0
    agrGB = 0
    agrGE = 0
    agrK = 0
    agrAG = 0

    bigrLU = 0
    bigrR = 0
    bigrKJ = 0
    bigrS = 0
    bigrGB = 0
    bigrGE = 0
    bigrK = 0
    bigrAG = 0

    trigrLU = 0
    trigrR = 0
    trigrKJ = 0
    trigrS = 0
    trigrGB = 0
    trigrGE = 0
    trigrK = 0
    trigrAG = 0

    quadrogrLU = 0
    quadrogrR = 0
    quadrogrKJ = 0
    quadrogrS = 0
    quadrogrGB = 0
    quadrogrGE = 0
    quadrogrK = 0
    quadrogrAG = 0

    pentagrLU = 0
    pentagrR = 0
    pentagrKJ = 0
    pentagrS = 0
    pentagrGB = 0
    pentagrGE = 0
    pentagrK = 0
    pentagrAG = 0

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

        try:
            biWord1 = str(textTockens[textTockens.index(i) - 1] + ' ' + word)
        except:
            biWord1 = ''
        try:
            biWord2 = str(word + ' ' + textTockens[textTockens.index(i) + 1])
        except:
            biWord2 = ''

        try:
            triWord1 = str(textTockens[textTockens.index(i) - 1] + ' ' + word + ' ' + textTockens[textTockens.index(i) + 1])
        except:
            triWord1 = ''
        try:
            triWord2 = str(textTockens[textTockens.index(i) - 2] + ' ' + textTockens[textTockens.index(i) - 1] + ' ' + word)
        except:
            triWord2 = ''
        try:
            triWord3 = str(word + ' ' + textTockens[textTockens.index(i) + 1] + ' ' + textTockens[textTockens.index(i) + 2])
        except:
            triWord3 = ''

        try:
            quadroWord1 = str(textTockens[textTockens.index(i) - 1] + ' ' + word + ' ' + textTockens[textTockens.index(i) + 1] + ' ' + textTockens[textTockens.index(i) + 2])
        except:
            quadroWord1 = ''
        try:
            quadroWord2 = str(textTockens[textTockens.index(i) - 2] + ' ' + textTockens[textTockens.index(i) - 1] + ' ' + word + ' ' + textTockens[textTockens.index(i) + 1])
        except:
            quadroWord2 = ''
        try:
            quadroWord1 = str(textTockens[textTockens.index(i) - 3] + ' ' + textTockens[textTockens.index(i) - 2] + ' ' + textTockens[textTockens.index(i) - 1] + ' ' + word)
        except:
            quadroWord3 = ''
        try:
            quadroWord2 = str(word + ' ' + textTockens[textTockens.index(i) + 1]  + ' ' + textTockens[textTockens.index(i) + 2]  + ' ' + textTockens[textTockens.index(i) + 3])
        except:
            quadroWord4 = ''

        try:
            pentaWord1 = str(textTockens[textTockens.index(i) - 3] + ' ' + textTockens[textTockens.index(i) - 2] + ' ' + textTockens[textTockens.index(i) - 1] + ' ' + word + ' ' + textTockens[textTockens.index(i) + 1])
        except:
            pentaWord1 = ''
        try:
            pentaWord2 = str(textTockens[textTockens.index(i) - 4] + ' ' + textTockens[textTockens.index(i) - 3] + ' ' + textTockens[textTockens.index(i) - 2] + ' ' + textTockens[textTockens.index(i) - 1] + ' ' + word)
        except:
            pentaWord2 = ''
        try:
            pentaWord1 = str(textTockens[textTockens.index(i) - 2] + ' ' + textTockens[textTockens.index(i) - 1] + ' ' + word + ' ' + textTockens[textTockens.index(i) + 1] + ' ' + textTockens[textTockens.index(i) + 2])
        except:
            pentaWord3 = ''
        try:
            pentaWord2 = str(textTockens[textTockens.index(i) - 1] + ' ' + word + ' ' + textTockens[textTockens.index(i) + 1] + ' ' + textTockens[textTockens.index(i) + 2] + ' ' + textTockens[textTockens.index(i) + 3])
        except:
            pentaWord4 = ''
        try:
            pentaWord1 = str(word + ' ' + textTockens[textTockens.index(i) + 1] + ' ' + textTockens[textTockens.index(i) + 2] + ' ' + textTockens[textTockens.index(i) + 3] + ' ' + textTockens[textTockens.index(i) + 4])
        except:
            pentaWord5 = ''

        biWords = [biWord1, biWord2]
        triWords = [triWord1, triWord2, triWord3]
        quadroWords = [quadroWord1, quadroWord2, quadroWord3, quadroWord4]
        pentaWords = [pentaWord1, pentaWord2, pentaWord3, pentaWord4, pentaWord5]

        for biWord in biWords:
            if biWord in dictLU:
                grLU += dictLU[biWord]
                allHits += 1
            if biWord in dictR:
                grR += dictR[biWord]
                allHits += 1
            if biWord in dictKJ:
                grKJ += dictKJ[biWord]
                allHits += 1
            if biWord in dictS:
                grS += dictS[biWord]
                allHits += 1
            if biWord in dictGB:
                grGB += dictGB[biWord]
                allHits += 1
            if biWord in dictGE:
                grGE += dictGE[biWord]
                allHits += 1
            if biWord in dictK:
                grK += dictK[biWord]
                allHits += 1
            if biWord in dictAG:
                grAG += dictAG[biWord]
                allHits += 1

        for triWord in triWords:
            if triWord in dictLU:
                grLU += dictLU[triWord]
                allHits += 1
            if triWord in dictR:
                grR += dictR[triWord]
                allHits += 1
            if triWord in dictKJ:
                grKJ += dictKJ[triWord]
                allHits += 1
            if triWord in dictS:
                grS += dictS[triWord]
                allHits += 1
            if triWord in dictGB:
                grGB += dictGB[triWord]
                allHits += 1
            if triWord in dictGE:
                grGE += dictGE[triWord]
                allHits += 1
            if triWord in dictK:
                grK += dictK[triWord]
                allHits += 1
            if triWord in dictAG:
                grAG += dictAG[triWord]
                allHits += 1

        for quadroWord in quadroWords:
            if quadroWord in dictLU:
                grLU += dictLU[quadroWord]
                allHits += 1
            if quadroWord in dictR:
                grR += dictR[quadroWord]
                allHits += 1
            if quadroWord in dictKJ:
                grKJ += dictKJ[quadroWord]
                allHits += 1
            if quadroWord in dictS:
                grS += dictS[quadroWord]
                allHits += 1
            if quadroWord in dictGB:
                grGB += dictGB[quadroWord]
                allHits += 1
            if quadroWord in dictGE:
                grGE += dictGE[quadroWord]
                allHits += 1
            if quadroWord in dictK:
                grK += dictK[quadroWord]
                allHits += 1
            if quadroWord in dictAG:
                grAG += dictAG[quadroWord]
                allHits += 1

        for pentaWord in pentaWords:
            if pentaWord in dictLU:
                grLU += dictLU[pentaWord]
                allHits += 1
            if pentaWord in dictR:
                grR += dictR[pentaWord]
                allHits += 1
            if pentaWord in dictKJ:
                grKJ += dictKJ[pentaWord]
                allHits += 1
            if pentaWord in dictS:
                grS += dictS[pentaWord]
                allHits += 1
            if pentaWord in dictGB:
                grGB += dictGB[pentaWord]
                allHits += 1
            if pentaWord in dictGE:
                grGE += dictGE[pentaWord]
                allHits += 1
            if pentaWord in dictK:
                grK += dictK[pentaWord]
                allHits += 1
            if pentaWord in dictAG:
                grAG += dictAG[pentaWord]
                allHits += 1

    for i in authors:
        if i in auDictLU:
            agrLU += auDictLU[i]
            allAuHits += 1
        if i in auDictR:
            agrR += auDictR[i]
            allAuHits += 1
        if i in auDictKJ:
            agrKJ += auDictKJ[i]
            allAuHits += 1
        if i in auDictS:
            agrS += auDictS[i]
            allAuHits += 1
        if i in auDictGB:
            agrGB += auDictGB[i]
            allAuHits += 1
        if i in auDictGE:
            agrGE += auDictGE[i]
            allAuHits += 1
        if i in auDictK:
            agrK += auDictK[i]
            allAuHits += 1
        if i in auDictAG:
            agrAG += auDictAG[i]
            allAuHits += 1

    if tockens == 0:
        tockens = 1

    rNouns = nNouns / tockens
    rVerbs = nVerbs / tockens
    rAdjectives = nAdjectives / tockens

    if allHits == 0:
        allHits = 1
    if allBiHits == 0:
        allBiHits = 1
    if allTriHits == 0:
        allTriHits = 1
    if allQuadroHits == 0:
        allQuadroHits = 1
    if allPentaHits == 0:
        allPentaHits = 1
    if allAuHits == 0:
        allAuHits = 1

    gdrLU = grLU / allHits
    gdrR = grR / allHits
    gdrKJ = grKJ / allHits
    gdrS = grS / allHits
    gdrGB = grGB / allHits
    gdrGE = grGE / allHits
    gdrK = grK / allHits
    gdrAG = grAG / allHits

    bigdrLU = bigrLU / allBiHits
    bigdrR = bigrR / allBiHits
    bigdrKJ = bigrKJ / allBiHits
    bigdrS = bigrS / allBiHits
    bigdrGB = bigrGB / allBiHits
    bigdrGE = bigrGE / allBiHits
    bigdrK = bigrK / allBiHits
    bigdrAG = bigrAG / allBiHits

    trigdrLU = trigrLU / allTriHits
    trigdrR = trigrR / allTriHits
    trigdrKJ = trigrKJ / allTriHits
    trigdrS = trigrS / allTriHits
    trigdrGB = trigrGB / allTriHits
    trigdrGE = trigrGE / allTriHits
    trigdrK = trigrK / allTriHits
    trigdrAG = trigrAG / allTriHits

    quadrogdrLU = quadrogrLU / allQuadroHits
    quadrogdrR = quadrogrR / allQuadroHits
    quadrogdrKJ = quadrogrKJ / allQuadroHits
    quadrogdrS = quadrogrS / allQuadroHits
    quadrogdrGB = quadrogrGB / allQuadroHits
    quadrogdrGE = quadrogrGE / allQuadroHits
    quadrogdrK = quadrogrK / allQuadroHits
    quadrogdrAG = quadrogrAG / allQuadroHits

    pentagdrLU = pentagrLU / allPentaHits
    pentagdrR = pentagrR / allPentaHits
    pentagdrKJ = pentagrKJ / allPentaHits
    pentagdrS = pentagrS / allPentaHits
    pentagdrGB = pentagrGB / allPentaHits
    pentagdrGE = pentagrGE / allPentaHits
    pentagdrK = pentagrK / allPentaHits
    pentagdrAG = pentagrAG / allPentaHits

    augdrLU = agrLU / allAuHits
    augdrR = agrR / allAuHits
    augdrKJ = agrKJ / allAuHits
    augdrS = agrS / allAuHits
    augdrGB = agrGB / allAuHits
    augdrGE = agrGE / allAuHits
    augdrK = agrK / allAuHits
    augdrAG = agrAG / allAuHits

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

    rCharCount = rCharCount / j2

    return [j, k, rNouns, rVerbs, rAdjectives, nCommas, nSymE, nSymH, nSymD, nSymP, nSymPa, nSymA, nSymS, nSymQ, nSymDa, nSymDd, nSymSc, gdrLU, gdrR, gdrKJ, gdrS, gdrGB, gdrGE, gdrK, gdrAG], [augdrLU, augdrR, augdrKJ, augdrS, augdrGB, augdrGE, augdrK, augdrAG], [bigdrLU, bigdrR, bigdrKJ, bigdrS, bigdrGB, bigdrGE, bigdrK, bigdrAG, rCharCount, maxCharCount, trigdrLU, trigdrR, trigdrKJ, trigdrS, trigdrGB, trigdrGE, trigdrK, trigdrAG, quadrogdrLU, quadrogdrR, quadrogdrKJ, quadrogdrS, quadrogdrGB, quadrogdrGE, quadrogdrK, quadrogdrAG, pentagdrLU, pentagdrR, pentagdrKJ, pentagdrS, pentagdrGB, pentagdrGE, pentagdrK, pentagdrAG]

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

    return meanLU, meanR, meanKJ, meanS, meanGB, meanGE, meanK, meanAG

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
        helpp.append(min(meanDistLU[i], meanDistR[i], meanDistKJ[i], meanDistS[i], meanDistGB[i], meanDistGE[i], meanDistK[i], meanDistAG[i]))

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
    global authorRatesTrain
    global biGenreRatesTrain
    """
    Erstellung eines Array mit den Featuren und Genre,  zur Uebergabe an createDataFrame()
    """

    currPos = 0
    # Creation of TrainDataFrame
    if multiprocessing:
        with Pool(4) as pool:
            data = pool.map(featurize, bookArray)
        list(data)
    else:
        for _ in bookArray:
            features, authorRates, biGenreRates = featurize(bookArray[currPos])
            authorRatesTrain.append(authorRates)
            biGenreRatesTrain.append(biGenreRates)
            features.append(bookArray[currPos][5])
            data.append(features)
            currPos += 1

def createTestDataArray():
    global currPos
    global tdata
    global tbookArray
    global authorRatesTest
    global biGenreRatesTest
    # Creation of TestDataFrame
    currPos = 0
    if multiprocessing:
        with Pool(4) as pool:
            tdata = pool.map(featurize, tbookArray)
        list(tdata)
    else:
        for _ in tbookArray:
            features, authorRates, biGenreRates = featurize(tbookArray[currPos])
            authorRatesTest.append(authorRates)
            biGenreRatesTest.append(biGenreRates)
            features.append(tbookArray[currPos][5])
            tdata.append(features)
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
    Erstellung der DataFrames und aufteilen in X_train, X_test(Features) und y_train, y_test(Genres)
    """
    # TrainDataFrame
    dataFrame=pd.DataFrame(data, columns=['WordsPerSentence', 'NumberSentences', 'PercentageNouns', 'PercentageVerbs', 'PercentageAdjectives', 'NumberCommas', 'NumberSymbols€', 'NumberSymbolsH', 'NumberSymbolsD', 'NumberSymbols%', 'NumberSymbols§', 'NumberSymbols&', 'NumberSymbols*', 'NumberSymbolsQ', 'NumberSymbols-', 'NumberSymbols:', 'NumberSymbols;', 'GenreRateLU', 'GenreRateR', 'GenreRateKJ', 'GenreRateS', 'GenreRateGB', 'GenreRateGE', 'GenreRateK', 'GenreRateAG', 'Genre', 'LUdistToGenreWordsPerSentence', 'LUdistToGenreNumberSentences', 'LUdistToGenrePercentageNouns', 'LUdistToGenrePercentageVerbs', 'LUdistToGenrePercentageAdjectives', 'LUdistToGenreNumberCommas', 'LUdistToGenreNumberSymbols€', 'LUdistToGenreNumberSymbolsH', 'LUdistToGenreNumberSymbolsD', 'LUdistToGenreNumberSymbols%', 'LUdistToGenreNumberSymbols§', 'LUdistToGenreNumberSymbols&', 'LUdistToGenreNumberSymbols*', 'LUdistToGenreNumberSymbolsQ', 'LUdistToGenreNumberSymbols-', 'LUdistToGenreNumberSymbols:', 'LUdistToGenreNumberSymbols;', 'LUdistToGenreGenreRateLU', 'LUdistToGenreGenreRateR', 'LUdistToGenreGenreRateKJ', 'LUdistToGenreGenreRateS', 'LUdistToGenreGenreRateGB', 'LUdistToGenreGenreRateGE', 'LUdistToGenreGenreRateK', 'LUdistToGenreGenreRateAG', 'RdistToGenreWordsPerSentence', 'RdistToGenreNumberSentences', 'RdistToGenrePercentageNouns', 'RdistToGenrePercentageVerbs', 'RdistToGenrePercentageAdjectives', 'RdistToGenreNumberCommas', 'RdistToGenreNumberSymbols€', 'RdistToGenreNumberSymbolsH', 'RdistToGenreNumberSymbolsD', 'RdistToGenreNumberSymbols%', 'RdistToGenreNumberSymbols§', 'RdistToGenreNumberSymbols&', 'RdistToGenreNumberSymbols*', 'RdistToGenreNumberSymbolsQ', 'RdistToGenreNumberSymbols-', 'RdistToGenreNumberSymbols:', 'RdistToGenreNumberSymbols;', 'RdistToGenreGenreRateLU', 'RdistToGenreGenreRateR', 'RdistToGenreGenreRateKJ', 'RdistToGenreGenreRateS', 'RdistToGenreGenreRateGB', 'RdistToGenreGenreRateGE', 'RdistToGenreGenreRateK', 'RdistToGenreGenreRateAG', 'KJdistToGenreWordsPerSentence', 'KJdistToGenreNumberSentences', 'KJdistToGenrePercentageNouns', 'KJdistToGenrePercentageVerbs', 'KJdistToGenrePercentageAdjectives', 'KJdistToGenreNumberCommas', 'KJdistToGenreNumberSymbols€', 'KJdistToGenreNumberSymbolsH', 'KJdistToGenreNumberSymbolsD', 'KJdistToGenreNumberSymbols%', 'KJdistToGenreNumberSymbols§', 'KJdistToGenreNumberSymbols&', 'KJdistToGenreNumberSymbols*', 'KJdistToGenreNumberSymbolsQ', 'KJdistToGenreNumberSymbols-', 'KJdistToGenreNumberSymbols:', 'KJdistToGenreNumberSymbols;', 'KJdistToGenreGenreRateLU', 'KJdistToGenreGenreRateR', 'KJdistToGenreGenreRateKJ', 'KJdistToGenreGenreRateS', 'KJdistToGenreGenreRateGB', 'KJdistToGenreGenreRateGE', 'KJdistToGenreGenreRateK', 'KJdistToGenreGenreRateAG', 'SdistToGenreWordsPerSentence', 'SdistToGenreNumberSentences', 'SdistToGenrePercentageNouns', 'SdistToGenrePercentageVerbs', 'SdistToGenrePercentageAdjectives', 'SdistToGenreNumberCommas', 'SdistToGenreNumberSymbols€', 'SdistToGenreNumberSymbolsH', 'SdistToGenreNumberSymbolsD', 'SdistToGenreNumberSymbols%', 'SdistToGenreNumberSymbols§', 'SdistToGenreNumberSymbols&', 'SdistToGenreNumberSymbols*', 'SdistToGenreNumberSymbolsQ', 'SdistToGenreNumberSymbols-', 'SdistToGenreNumberSymbols:', 'SdistToGenreNumberSymbols;', 'SdistToGenreGenreRateLU', 'SdistToGenreGenreRateR', 'SdistToGenreGenreRateKJ', 'SdistToGenreGenreRateS', 'SdistToGenreGenreRateGB', 'SdistToGenreGenreRateGE', 'SdistToGenreGenreRateK', 'SdistToGenreGenreRateAG', 'GBdistToGenreWordsPerSentence', 'GBdistToGenreNumberSentences', 'GBdistToGenrePercentageNouns', 'GBdistToGenrePercentageVerbs', 'GBdistToGenrePercentageAdjectives', 'GBdistToGenreNumberCommas', 'GBdistToGenreNumberSymbols€', 'GBdistToGenreNumberSymbolsH', 'GBdistToGenreNumberSymbolsD', 'GBdistToGenreNumberSymbols%', 'GBdistToGenreNumberSymbols§', 'GBdistToGenreNumberSymbols&', 'GBdistToGenreNumberSymbols*', 'GBdistToGenreNumberSymbolsQ', 'GBdistToGenreNumberSymbols-', 'GBdistToGenreNumberSymbols:', 'GBdistToGenreNumberSymbols;', 'GBdistToGenreGenreRateLU', 'GBdistToGenreGenreRateR', 'GBdistToGenreGenreRateKJ', 'GBdistToGenreGenreRateS', 'GBdistToGenreGenreRateGB', 'GBdistToGenreGenreRateGE', 'GBdistToGenreGenreRateK', 'GBdistToGenreGenreRateAG', 'GEdistToGenreWordsPerSentence', 'GEdistToGenreNumberSentences', 'GEdistToGenrePercentageNouns', 'GEdistToGenrePercentageVerbs', 'GEdistToGenrePercentageAdjectives', 'GEdistToGenreNumberCommas', 'GEdistToGenreNumberSymbols€', 'GEdistToGenreNumberSymbolsH', 'GEdistToGenreNumberSymbolsD', 'GEdistToGenreNumberSymbols%', 'GEdistToGenreNumberSymbols§', 'GEdistToGenreNumberSymbols&', 'GEdistToGenreNumberSymbols*', 'GEdistToGenreNumberSymbolsQ', 'GEdistToGenreNumberSymbols-', 'GEdistToGenreNumberSymbols:', 'GEdistToGenreNumberSymbols;', 'GEdistToGenreGenreRateLU', 'GEdistToGenreGenreRateR', 'GEdistToGenreGenreRateKJ', 'GEdistToGenreGenreRateS', 'GEdistToGenreGenreRateGB', 'GEdistToGenreGenreRateGE', 'GEdistToGenreGenreRateK', 'GEdistToGenreGenreRateAG', 'KdistToGenreWordsPerSentence', 'KdistToGenreNumberSentences', 'KdistToGenrePercentageNouns', 'KdistToGenrePercentageVerbs', 'KdistToGenrePercentageAdjectives', 'KdistToGenreNumberCommas', 'KdistToGenreNumberSymbols€', 'KdistToGenreNumberSymbolsH', 'KdistToGenreNumberSymbolsD', 'KdistToGenreNumberSymbols%', 'KdistToGenreNumberSymbols§', 'KdistToGenreNumberSymbols&', 'KdistToGenreNumberSymbols*', 'KdistToGenreNumberSymbolsQ', 'KdistToGenreNumberSymbols-', 'KdistToGenreNumberSymbols:', 'KdistToGenreNumberSymbols;', 'KdistToGenreGenreRateLU', 'KdistToGenreGenreRateR', 'KdistToGenreGenreRateKJ', 'KdistToGenreGenreRateS', 'KdistToGenreGenreRateGB', 'KdistToGenreGenreRateGE', 'KdistToGenreGenreRateK', 'KdistToGenreGenreRateAG', 'AGdistToGenreWordsPerSentence', 'AGdistToGenreNumberSentences', 'AGdistToGenrePercentageNouns', 'AGdistToGenrePercentageVerbs', 'AGdistToGenrePercentageAdjectives', 'AGdistToGenreNumberCommas', 'AGdistToGenreNumberSymbols€', 'AGdistToGenreNumberSymbolsH', 'AGdistToGenreNumberSymbolsD', 'AGdistToGenreNumberSymbols%', 'AGdistToGenreNumberSymbols§', 'AGdistToGenreNumberSymbols&', 'AGdistToGenreNumberSymbols*', 'AGdistToGenreNumberSymbolsQ', 'AGdistToGenreNumberSymbols-', 'AGdistToGenreNumberSymbols:', 'AGdistToGenreNumberSymbols;', 'AGdistToGenreGenreRateLU', 'AGdistToGenreGenreRateR', 'AGdistToGenreGenreRateKJ', 'AGdistToGenreGenreRateS', 'AGdistToGenreGenreRateGB', 'AGdistToGenreGenreRateGE', 'AGdistToGenreGenreRateK', 'AGdistToGenreGenreRateAG', 'minDistToGenreWordsPerSentence', 'minDistToGenreNumberSentences', 'minDistToGenrePercentageNouns', 'minDistToGenrePercentageVerbs', 'minDistToGenrePercentageAdjectives', 'minDistToGenreNumberCommas', 'minDistToGenreNumberSymbols€', 'minDistToGenreNumberSymbolsH', 'minDistToGenreNumberSymbolsD', 'minDistToGenreNumberSymbols%', 'minDistToGenreNumberSymbols§', 'minDistToGenreNumberSymbols&', 'minDistToGenreNumberSymbols*', 'minDistToGenreNumberSymbolsQ', 'minDistToGenreNumberSymbols-', 'minDistToGenreNumberSymbols:', 'minDistToGenreNumberSymbols;', 'minDistToGenreGenreRateLU', 'minDistToGenreGenreRateR', 'minDistToGenreGenreRateKJ', 'minDistToGenreGenreRateS', 'minDistToGenreGenreRateGB', 'minDistToGenreGenreRateGE', 'minDistToGenreGenreRateK', 'minDistToGenreGenreRateAG', 'AuthorGenreRateLU', 'AuthorGenreRateR', 'AuthorGenreRateKJ', 'AuthorGenreRateS', 'AuthorGenreRateGB', 'AuthorGenreRateGE', 'AuthorGenreRateK', 'AuthorGenreRateAG', 'BiGramGenreRateLU', 'BiGramGenreRateR', 'BiGramGenreRateKJ', 'BiGramGenreRateS', 'BiGramGenreRateGB', 'BiGramGenreRateGE', 'BiGramGenreRateK', 'BiGramGenreRateAG', 'CharsPerWord', 'MaxCharCount', 'TriGramGenreRateLU', 'TriGramGenreRateR', 'TriGramGenreRateKJ', 'TriGramGenreRateS', 'TriGramGenreRateGB', 'TriGramGenreRateGE', 'TriGramGenreRateK', 'TriGramGenreRateAG', 'QuadroGramGenreRateLU', 'QuadroGramGenreRateR', 'QuadroGramGenreRateKJ', 'QuadroGramGenreRateS', 'QuadroGramGenreRateGB', 'QuadroGramGenreRateGE', 'QuadroGramGenreRateK', 'QuadroGramGenreRateAG', 'PentaGramGenreRateLU', 'PentaGramGenreRateR', 'PentaGramGenreRateKJ', 'PentaGramGenreRateS', 'PentaGramGenreRateGB', 'PentaGramGenreRateGE', 'PentaGramGenreRateK', 'PentaGramGenreRateAG'], dtype=float)

    # TestDataFrame
    tdataFrame=pd.DataFrame(tdata, columns=['WordsPerSentence', 'NumberSentences', 'PercentageNouns', 'PercentageVerbs', 'PercentageAdjectives', 'NumberCommas', 'NumberSymbols€', 'NumberSymbolsH', 'NumberSymbolsD', 'NumberSymbols%', 'NumberSymbols§', 'NumberSymbols&', 'NumberSymbols*', 'NumberSymbolsQ', 'NumberSymbols-', 'NumberSymbols:', 'NumberSymbols;', 'GenreRateLU', 'GenreRateR', 'GenreRateKJ', 'GenreRateS', 'GenreRateGB', 'GenreRateGE', 'GenreRateK', 'GenreRateAG', 'Genre', 'LUdistToGenreWordsPerSentence', 'LUdistToGenreNumberSentences', 'LUdistToGenrePercentageNouns', 'LUdistToGenrePercentageVerbs', 'LUdistToGenrePercentageAdjectives', 'LUdistToGenreNumberCommas', 'LUdistToGenreNumberSymbols€', 'LUdistToGenreNumberSymbolsH', 'LUdistToGenreNumberSymbolsD', 'LUdistToGenreNumberSymbols%', 'LUdistToGenreNumberSymbols§', 'LUdistToGenreNumberSymbols&', 'LUdistToGenreNumberSymbols*', 'LUdistToGenreNumberSymbolsQ', 'LUdistToGenreNumberSymbols-', 'LUdistToGenreNumberSymbols:', 'LUdistToGenreNumberSymbols;', 'LUdistToGenreGenreRateLU', 'LUdistToGenreGenreRateR', 'LUdistToGenreGenreRateKJ', 'LUdistToGenreGenreRateS', 'LUdistToGenreGenreRateGB', 'LUdistToGenreGenreRateGE', 'LUdistToGenreGenreRateK', 'LUdistToGenreGenreRateAG', 'RdistToGenreWordsPerSentence', 'RdistToGenreNumberSentences', 'RdistToGenrePercentageNouns', 'RdistToGenrePercentageVerbs', 'RdistToGenrePercentageAdjectives', 'RdistToGenreNumberCommas', 'RdistToGenreNumberSymbols€', 'RdistToGenreNumberSymbolsH', 'RdistToGenreNumberSymbolsD', 'RdistToGenreNumberSymbols%', 'RdistToGenreNumberSymbols§', 'RdistToGenreNumberSymbols&', 'RdistToGenreNumberSymbols*', 'RdistToGenreNumberSymbolsQ', 'RdistToGenreNumberSymbols-', 'RdistToGenreNumberSymbols:', 'RdistToGenreNumberSymbols;', 'RdistToGenreGenreRateLU', 'RdistToGenreGenreRateR', 'RdistToGenreGenreRateKJ', 'RdistToGenreGenreRateS', 'RdistToGenreGenreRateGB', 'RdistToGenreGenreRateGE', 'RdistToGenreGenreRateK', 'RdistToGenreGenreRateAG', 'KJdistToGenreWordsPerSentence', 'KJdistToGenreNumberSentences', 'KJdistToGenrePercentageNouns', 'KJdistToGenrePercentageVerbs', 'KJdistToGenrePercentageAdjectives', 'KJdistToGenreNumberCommas', 'KJdistToGenreNumberSymbols€', 'KJdistToGenreNumberSymbolsH', 'KJdistToGenreNumberSymbolsD', 'KJdistToGenreNumberSymbols%', 'KJdistToGenreNumberSymbols§', 'KJdistToGenreNumberSymbols&', 'KJdistToGenreNumberSymbols*', 'KJdistToGenreNumberSymbolsQ', 'KJdistToGenreNumberSymbols-', 'KJdistToGenreNumberSymbols:', 'KJdistToGenreNumberSymbols;', 'KJdistToGenreGenreRateLU', 'KJdistToGenreGenreRateR', 'KJdistToGenreGenreRateKJ', 'KJdistToGenreGenreRateS', 'KJdistToGenreGenreRateGB', 'KJdistToGenreGenreRateGE', 'KJdistToGenreGenreRateK', 'KJdistToGenreGenreRateAG', 'SdistToGenreWordsPerSentence', 'SdistToGenreNumberSentences', 'SdistToGenrePercentageNouns', 'SdistToGenrePercentageVerbs', 'SdistToGenrePercentageAdjectives', 'SdistToGenreNumberCommas', 'SdistToGenreNumberSymbols€', 'SdistToGenreNumberSymbolsH', 'SdistToGenreNumberSymbolsD', 'SdistToGenreNumberSymbols%', 'SdistToGenreNumberSymbols§', 'SdistToGenreNumberSymbols&', 'SdistToGenreNumberSymbols*', 'SdistToGenreNumberSymbolsQ', 'SdistToGenreNumberSymbols-', 'SdistToGenreNumberSymbols:', 'SdistToGenreNumberSymbols;', 'SdistToGenreGenreRateLU', 'SdistToGenreGenreRateR', 'SdistToGenreGenreRateKJ', 'SdistToGenreGenreRateS', 'SdistToGenreGenreRateGB', 'SdistToGenreGenreRateGE', 'SdistToGenreGenreRateK', 'SdistToGenreGenreRateAG', 'GBdistToGenreWordsPerSentence', 'GBdistToGenreNumberSentences', 'GBdistToGenrePercentageNouns', 'GBdistToGenrePercentageVerbs', 'GBdistToGenrePercentageAdjectives', 'GBdistToGenreNumberCommas', 'GBdistToGenreNumberSymbols€', 'GBdistToGenreNumberSymbolsH', 'GBdistToGenreNumberSymbolsD', 'GBdistToGenreNumberSymbols%', 'GBdistToGenreNumberSymbols§', 'GBdistToGenreNumberSymbols&', 'GBdistToGenreNumberSymbols*', 'GBdistToGenreNumberSymbolsQ', 'GBdistToGenreNumberSymbols-', 'GBdistToGenreNumberSymbols:', 'GBdistToGenreNumberSymbols;', 'GBdistToGenreGenreRateLU', 'GBdistToGenreGenreRateR', 'GBdistToGenreGenreRateKJ', 'GBdistToGenreGenreRateS', 'GBdistToGenreGenreRateGB', 'GBdistToGenreGenreRateGE', 'GBdistToGenreGenreRateK', 'GBdistToGenreGenreRateAG', 'GEdistToGenreWordsPerSentence', 'GEdistToGenreNumberSentences', 'GEdistToGenrePercentageNouns', 'GEdistToGenrePercentageVerbs', 'GEdistToGenrePercentageAdjectives', 'GEdistToGenreNumberCommas', 'GEdistToGenreNumberSymbols€', 'GEdistToGenreNumberSymbolsH', 'GEdistToGenreNumberSymbolsD', 'GEdistToGenreNumberSymbols%', 'GEdistToGenreNumberSymbols§', 'GEdistToGenreNumberSymbols&', 'GEdistToGenreNumberSymbols*', 'GEdistToGenreNumberSymbolsQ', 'GEdistToGenreNumberSymbols-', 'GEdistToGenreNumberSymbols:', 'GEdistToGenreNumberSymbols;', 'GEdistToGenreGenreRateLU', 'GEdistToGenreGenreRateR', 'GEdistToGenreGenreRateKJ', 'GEdistToGenreGenreRateS', 'GEdistToGenreGenreRateGB', 'GEdistToGenreGenreRateGE', 'GEdistToGenreGenreRateK', 'GEdistToGenreGenreRateAG', 'KdistToGenreWordsPerSentence', 'KdistToGenreNumberSentences', 'KdistToGenrePercentageNouns', 'KdistToGenrePercentageVerbs', 'KdistToGenrePercentageAdjectives', 'KdistToGenreNumberCommas', 'KdistToGenreNumberSymbols€', 'KdistToGenreNumberSymbolsH', 'KdistToGenreNumberSymbolsD', 'KdistToGenreNumberSymbols%', 'KdistToGenreNumberSymbols§', 'KdistToGenreNumberSymbols&', 'KdistToGenreNumberSymbols*', 'KdistToGenreNumberSymbolsQ', 'KdistToGenreNumberSymbols-', 'KdistToGenreNumberSymbols:', 'KdistToGenreNumberSymbols;', 'KdistToGenreGenreRateLU', 'KdistToGenreGenreRateR', 'KdistToGenreGenreRateKJ', 'KdistToGenreGenreRateS', 'KdistToGenreGenreRateGB', 'KdistToGenreGenreRateGE', 'KdistToGenreGenreRateK', 'KdistToGenreGenreRateAG', 'AGdistToGenreWordsPerSentence', 'AGdistToGenreNumberSentences', 'AGdistToGenrePercentageNouns', 'AGdistToGenrePercentageVerbs', 'AGdistToGenrePercentageAdjectives', 'AGdistToGenreNumberCommas', 'AGdistToGenreNumberSymbols€', 'AGdistToGenreNumberSymbolsH', 'AGdistToGenreNumberSymbolsD', 'AGdistToGenreNumberSymbols%', 'AGdistToGenreNumberSymbols§', 'AGdistToGenreNumberSymbols&', 'AGdistToGenreNumberSymbols*', 'AGdistToGenreNumberSymbolsQ', 'AGdistToGenreNumberSymbols-', 'AGdistToGenreNumberSymbols:', 'AGdistToGenreNumberSymbols;', 'AGdistToGenreGenreRateLU', 'AGdistToGenreGenreRateR', 'AGdistToGenreGenreRateKJ', 'AGdistToGenreGenreRateS', 'AGdistToGenreGenreRateGB', 'AGdistToGenreGenreRateGE', 'AGdistToGenreGenreRateK', 'AGdistToGenreGenreRateAG', 'minDistToGenreWordsPerSentence', 'minDistToGenreNumberSentences', 'minDistToGenrePercentageNouns', 'minDistToGenrePercentageVerbs', 'minDistToGenrePercentageAdjectives', 'minDistToGenreNumberCommas', 'minDistToGenreNumberSymbols€', 'minDistToGenreNumberSymbolsH', 'minDistToGenreNumberSymbolsD', 'minDistToGenreNumberSymbols%', 'minDistToGenreNumberSymbols§', 'minDistToGenreNumberSymbols&', 'minDistToGenreNumberSymbols*', 'minDistToGenreNumberSymbolsQ', 'minDistToGenreNumberSymbols-', 'minDistToGenreNumberSymbols:', 'minDistToGenreNumberSymbols;', 'minDistToGenreGenreRateLU', 'minDistToGenreGenreRateR', 'minDistToGenreGenreRateKJ', 'minDistToGenreGenreRateS', 'minDistToGenreGenreRateGB', 'minDistToGenreGenreRateGE', 'minDistToGenreGenreRateK', 'minDistToGenreGenreRateAG', 'AuthorGenreRateLU', 'AuthorGenreRateR', 'AuthorGenreRateKJ', 'AuthorGenreRateS', 'AuthorGenreRateGB', 'AuthorGenreRateGE', 'AuthorGenreRateK', 'AuthorGenreRateAG', 'BiGramGenreRateLU', 'BiGramGenreRateR', 'BiGramGenreRateKJ', 'BiGramGenreRateS', 'BiGramGenreRateGB', 'BiGramGenreRateGE', 'BiGramGenreRateK', 'BiGramGenreRateAG', 'CharsPerWord', 'MaxCharCount', 'TriGramGenreRateLU', 'TriGramGenreRateR', 'TriGramGenreRateKJ', 'TriGramGenreRateS', 'TriGramGenreRateGB', 'TriGramGenreRateGE', 'TriGramGenreRateK', 'TriGramGenreRateAG', 'QuadroGramGenreRateLU', 'QuadroGramGenreRateR', 'QuadroGramGenreRateKJ', 'QuadroGramGenreRateS', 'QuadroGramGenreRateGB', 'QuadroGramGenreRateGE', 'QuadroGramGenreRateK', 'QuadroGramGenreRateAG', 'PentaGramGenreRateLU', 'PentaGramGenreRateR', 'PentaGramGenreRateKJ', 'PentaGramGenreRateS', 'PentaGramGenreRateGB', 'PentaGramGenreRateGE', 'PentaGramGenreRateK', 'PentaGramGenreRateAG'], dtype=float)

    # TrainData
    X_train=dataFrame[['WordsPerSentence', 'NumberSentences', 'PercentageNouns', 'PercentageVerbs', 'PercentageAdjectives', 'NumberCommas', 'NumberSymbols€', 'NumberSymbolsH', 'NumberSymbolsD', 'NumberSymbols%', 'NumberSymbols§', 'NumberSymbols&', 'NumberSymbols*', 'NumberSymbolsQ', 'NumberSymbols-', 'NumberSymbols:', 'NumberSymbols;', 'GenreRateLU', 'GenreRateR', 'GenreRateKJ', 'GenreRateS', 'GenreRateGB', 'GenreRateGE', 'GenreRateK', 'GenreRateAG', 'LUdistToGenreWordsPerSentence', 'LUdistToGenreNumberSentences', 'LUdistToGenrePercentageNouns', 'LUdistToGenrePercentageVerbs', 'LUdistToGenrePercentageAdjectives', 'LUdistToGenreNumberCommas', 'LUdistToGenreNumberSymbols€', 'LUdistToGenreNumberSymbolsH', 'LUdistToGenreNumberSymbolsD', 'LUdistToGenreNumberSymbols%', 'LUdistToGenreNumberSymbols§', 'LUdistToGenreNumberSymbols&', 'LUdistToGenreNumberSymbols*', 'LUdistToGenreNumberSymbolsQ', 'LUdistToGenreNumberSymbols-', 'LUdistToGenreNumberSymbols:', 'LUdistToGenreNumberSymbols;', 'LUdistToGenreGenreRateLU', 'LUdistToGenreGenreRateR', 'LUdistToGenreGenreRateKJ', 'LUdistToGenreGenreRateS', 'LUdistToGenreGenreRateGB', 'LUdistToGenreGenreRateGE', 'LUdistToGenreGenreRateK', 'LUdistToGenreGenreRateAG', 'RdistToGenreWordsPerSentence', 'RdistToGenreNumberSentences', 'RdistToGenrePercentageNouns', 'RdistToGenrePercentageVerbs', 'RdistToGenrePercentageAdjectives', 'RdistToGenreNumberCommas', 'RdistToGenreNumberSymbols€', 'RdistToGenreNumberSymbolsH', 'RdistToGenreNumberSymbolsD', 'RdistToGenreNumberSymbols%', 'RdistToGenreNumberSymbols§', 'RdistToGenreNumberSymbols&', 'RdistToGenreNumberSymbols*', 'RdistToGenreNumberSymbolsQ', 'RdistToGenreNumberSymbols-', 'RdistToGenreNumberSymbols:', 'RdistToGenreNumberSymbols;', 'RdistToGenreGenreRateLU', 'RdistToGenreGenreRateR', 'RdistToGenreGenreRateKJ', 'RdistToGenreGenreRateS', 'RdistToGenreGenreRateGB', 'RdistToGenreGenreRateGE', 'RdistToGenreGenreRateK', 'RdistToGenreGenreRateAG', 'KJdistToGenreWordsPerSentence', 'KJdistToGenreNumberSentences', 'KJdistToGenrePercentageNouns', 'KJdistToGenrePercentageVerbs', 'KJdistToGenrePercentageAdjectives', 'KJdistToGenreNumberCommas', 'KJdistToGenreNumberSymbols€', 'KJdistToGenreNumberSymbolsH', 'KJdistToGenreNumberSymbolsD', 'KJdistToGenreNumberSymbols%', 'KJdistToGenreNumberSymbols§', 'KJdistToGenreNumberSymbols&', 'KJdistToGenreNumberSymbols*', 'KJdistToGenreNumberSymbolsQ', 'KJdistToGenreNumberSymbols-', 'KJdistToGenreNumberSymbols:', 'KJdistToGenreNumberSymbols;', 'KJdistToGenreGenreRateLU', 'KJdistToGenreGenreRateR', 'KJdistToGenreGenreRateKJ', 'KJdistToGenreGenreRateS', 'KJdistToGenreGenreRateGB', 'KJdistToGenreGenreRateGE', 'KJdistToGenreGenreRateK', 'KJdistToGenreGenreRateAG', 'SdistToGenreWordsPerSentence', 'SdistToGenreNumberSentences', 'SdistToGenrePercentageNouns', 'SdistToGenrePercentageVerbs', 'SdistToGenrePercentageAdjectives', 'SdistToGenreNumberCommas', 'SdistToGenreNumberSymbols€', 'SdistToGenreNumberSymbolsH', 'SdistToGenreNumberSymbolsD', 'SdistToGenreNumberSymbols%', 'SdistToGenreNumberSymbols§', 'SdistToGenreNumberSymbols&', 'SdistToGenreNumberSymbols*', 'SdistToGenreNumberSymbolsQ', 'SdistToGenreNumberSymbols-', 'SdistToGenreNumberSymbols:', 'SdistToGenreNumberSymbols;', 'SdistToGenreGenreRateLU', 'SdistToGenreGenreRateR', 'SdistToGenreGenreRateKJ', 'SdistToGenreGenreRateS', 'SdistToGenreGenreRateGB', 'SdistToGenreGenreRateGE', 'SdistToGenreGenreRateK', 'SdistToGenreGenreRateAG', 'GBdistToGenreWordsPerSentence', 'GBdistToGenreNumberSentences', 'GBdistToGenrePercentageNouns', 'GBdistToGenrePercentageVerbs', 'GBdistToGenrePercentageAdjectives', 'GBdistToGenreNumberCommas', 'GBdistToGenreNumberSymbols€', 'GBdistToGenreNumberSymbolsH', 'GBdistToGenreNumberSymbolsD', 'GBdistToGenreNumberSymbols%', 'GBdistToGenreNumberSymbols§', 'GBdistToGenreNumberSymbols&', 'GBdistToGenreNumberSymbols*', 'GBdistToGenreNumberSymbolsQ', 'GBdistToGenreNumberSymbols-', 'GBdistToGenreNumberSymbols:', 'GBdistToGenreNumberSymbols;', 'GBdistToGenreGenreRateLU', 'GBdistToGenreGenreRateR', 'GBdistToGenreGenreRateKJ', 'GBdistToGenreGenreRateS', 'GBdistToGenreGenreRateGB', 'GBdistToGenreGenreRateGE', 'GBdistToGenreGenreRateK', 'GBdistToGenreGenreRateAG', 'GEdistToGenreWordsPerSentence', 'GEdistToGenreNumberSentences', 'GEdistToGenrePercentageNouns', 'GEdistToGenrePercentageVerbs', 'GEdistToGenrePercentageAdjectives', 'GEdistToGenreNumberCommas', 'GEdistToGenreNumberSymbols€', 'GEdistToGenreNumberSymbolsH', 'GEdistToGenreNumberSymbolsD', 'GEdistToGenreNumberSymbols%', 'GEdistToGenreNumberSymbols§', 'GEdistToGenreNumberSymbols&', 'GEdistToGenreNumberSymbols*', 'GEdistToGenreNumberSymbolsQ', 'GEdistToGenreNumberSymbols-', 'GEdistToGenreNumberSymbols:', 'GEdistToGenreNumberSymbols;', 'GEdistToGenreGenreRateLU', 'GEdistToGenreGenreRateR', 'GEdistToGenreGenreRateKJ', 'GEdistToGenreGenreRateS', 'GEdistToGenreGenreRateGB', 'GEdistToGenreGenreRateGE', 'GEdistToGenreGenreRateK', 'GEdistToGenreGenreRateAG', 'KdistToGenreWordsPerSentence', 'KdistToGenreNumberSentences', 'KdistToGenrePercentageNouns', 'KdistToGenrePercentageVerbs', 'KdistToGenrePercentageAdjectives', 'KdistToGenreNumberCommas', 'KdistToGenreNumberSymbols€', 'KdistToGenreNumberSymbolsH', 'KdistToGenreNumberSymbolsD', 'KdistToGenreNumberSymbols%', 'KdistToGenreNumberSymbols§', 'KdistToGenreNumberSymbols&', 'KdistToGenreNumberSymbols*', 'KdistToGenreNumberSymbolsQ', 'KdistToGenreNumberSymbols-', 'KdistToGenreNumberSymbols:', 'KdistToGenreNumberSymbols;', 'KdistToGenreGenreRateLU', 'KdistToGenreGenreRateR', 'KdistToGenreGenreRateKJ', 'KdistToGenreGenreRateS', 'KdistToGenreGenreRateGB', 'KdistToGenreGenreRateGE', 'KdistToGenreGenreRateK', 'KdistToGenreGenreRateAG', 'AGdistToGenreWordsPerSentence', 'AGdistToGenreNumberSentences', 'AGdistToGenrePercentageNouns', 'AGdistToGenrePercentageVerbs', 'AGdistToGenrePercentageAdjectives', 'AGdistToGenreNumberCommas', 'AGdistToGenreNumberSymbols€', 'AGdistToGenreNumberSymbolsH', 'AGdistToGenreNumberSymbolsD', 'AGdistToGenreNumberSymbols%', 'AGdistToGenreNumberSymbols§', 'AGdistToGenreNumberSymbols&', 'AGdistToGenreNumberSymbols*', 'AGdistToGenreNumberSymbolsQ', 'AGdistToGenreNumberSymbols-', 'AGdistToGenreNumberSymbols:', 'AGdistToGenreNumberSymbols;', 'AGdistToGenreGenreRateLU', 'AGdistToGenreGenreRateR', 'AGdistToGenreGenreRateKJ', 'AGdistToGenreGenreRateS', 'AGdistToGenreGenreRateGB', 'AGdistToGenreGenreRateGE', 'AGdistToGenreGenreRateK', 'AGdistToGenreGenreRateAG', 'minDistToGenreWordsPerSentence', 'minDistToGenreNumberSentences', 'minDistToGenrePercentageNouns', 'minDistToGenrePercentageVerbs', 'minDistToGenrePercentageAdjectives', 'minDistToGenreNumberCommas', 'minDistToGenreNumberSymbols€', 'minDistToGenreNumberSymbolsH', 'minDistToGenreNumberSymbolsD', 'minDistToGenreNumberSymbols%', 'minDistToGenreNumberSymbols§', 'minDistToGenreNumberSymbols&', 'minDistToGenreNumberSymbols*', 'minDistToGenreNumberSymbolsQ', 'minDistToGenreNumberSymbols-', 'minDistToGenreNumberSymbols:', 'minDistToGenreNumberSymbols;', 'minDistToGenreGenreRateLU', 'minDistToGenreGenreRateR', 'minDistToGenreGenreRateKJ', 'minDistToGenreGenreRateS', 'minDistToGenreGenreRateGB', 'minDistToGenreGenreRateGE', 'minDistToGenreGenreRateK', 'minDistToGenreGenreRateAG', 'AuthorGenreRateLU', 'AuthorGenreRateR', 'AuthorGenreRateKJ', 'AuthorGenreRateS', 'AuthorGenreRateGB', 'AuthorGenreRateGE', 'AuthorGenreRateK', 'AuthorGenreRateAG']]
    y_train=dataFrame['Genre']

    # TestData
    X_test=tdataFrame[['WordsPerSentence', 'NumberSentences', 'PercentageNouns', 'PercentageVerbs', 'PercentageAdjectives', 'NumberCommas', 'NumberSymbols€', 'NumberSymbolsH', 'NumberSymbolsD', 'NumberSymbols%', 'NumberSymbols§', 'NumberSymbols&', 'NumberSymbols*', 'NumberSymbolsQ', 'NumberSymbols-', 'NumberSymbols:', 'NumberSymbols;', 'GenreRateLU', 'GenreRateR', 'GenreRateKJ', 'GenreRateS', 'GenreRateGB', 'GenreRateGE', 'GenreRateK', 'GenreRateAG', 'LUdistToGenreWordsPerSentence', 'LUdistToGenreNumberSentences', 'LUdistToGenrePercentageNouns', 'LUdistToGenrePercentageVerbs', 'LUdistToGenrePercentageAdjectives', 'LUdistToGenreNumberCommas', 'LUdistToGenreNumberSymbols€', 'LUdistToGenreNumberSymbolsH', 'LUdistToGenreNumberSymbolsD', 'LUdistToGenreNumberSymbols%', 'LUdistToGenreNumberSymbols§', 'LUdistToGenreNumberSymbols&', 'LUdistToGenreNumberSymbols*', 'LUdistToGenreNumberSymbolsQ', 'LUdistToGenreNumberSymbols-', 'LUdistToGenreNumberSymbols:', 'LUdistToGenreNumberSymbols;', 'LUdistToGenreGenreRateLU', 'LUdistToGenreGenreRateR', 'LUdistToGenreGenreRateKJ', 'LUdistToGenreGenreRateS', 'LUdistToGenreGenreRateGB', 'LUdistToGenreGenreRateGE', 'LUdistToGenreGenreRateK', 'LUdistToGenreGenreRateAG', 'RdistToGenreWordsPerSentence', 'RdistToGenreNumberSentences', 'RdistToGenrePercentageNouns', 'RdistToGenrePercentageVerbs', 'RdistToGenrePercentageAdjectives', 'RdistToGenreNumberCommas', 'RdistToGenreNumberSymbols€', 'RdistToGenreNumberSymbolsH', 'RdistToGenreNumberSymbolsD', 'RdistToGenreNumberSymbols%', 'RdistToGenreNumberSymbols§', 'RdistToGenreNumberSymbols&', 'RdistToGenreNumberSymbols*', 'RdistToGenreNumberSymbolsQ', 'RdistToGenreNumberSymbols-', 'RdistToGenreNumberSymbols:', 'RdistToGenreNumberSymbols;', 'RdistToGenreGenreRateLU', 'RdistToGenreGenreRateR', 'RdistToGenreGenreRateKJ', 'RdistToGenreGenreRateS', 'RdistToGenreGenreRateGB', 'RdistToGenreGenreRateGE', 'RdistToGenreGenreRateK', 'RdistToGenreGenreRateAG', 'KJdistToGenreWordsPerSentence', 'KJdistToGenreNumberSentences', 'KJdistToGenrePercentageNouns', 'KJdistToGenrePercentageVerbs', 'KJdistToGenrePercentageAdjectives', 'KJdistToGenreNumberCommas', 'KJdistToGenreNumberSymbols€', 'KJdistToGenreNumberSymbolsH', 'KJdistToGenreNumberSymbolsD', 'KJdistToGenreNumberSymbols%', 'KJdistToGenreNumberSymbols§', 'KJdistToGenreNumberSymbols&', 'KJdistToGenreNumberSymbols*', 'KJdistToGenreNumberSymbolsQ', 'KJdistToGenreNumberSymbols-', 'KJdistToGenreNumberSymbols:', 'KJdistToGenreNumberSymbols;', 'KJdistToGenreGenreRateLU', 'KJdistToGenreGenreRateR', 'KJdistToGenreGenreRateKJ', 'KJdistToGenreGenreRateS', 'KJdistToGenreGenreRateGB', 'KJdistToGenreGenreRateGE', 'KJdistToGenreGenreRateK', 'KJdistToGenreGenreRateAG', 'SdistToGenreWordsPerSentence', 'SdistToGenreNumberSentences', 'SdistToGenrePercentageNouns', 'SdistToGenrePercentageVerbs', 'SdistToGenrePercentageAdjectives', 'SdistToGenreNumberCommas', 'SdistToGenreNumberSymbols€', 'SdistToGenreNumberSymbolsH', 'SdistToGenreNumberSymbolsD', 'SdistToGenreNumberSymbols%', 'SdistToGenreNumberSymbols§', 'SdistToGenreNumberSymbols&', 'SdistToGenreNumberSymbols*', 'SdistToGenreNumberSymbolsQ', 'SdistToGenreNumberSymbols-', 'SdistToGenreNumberSymbols:', 'SdistToGenreNumberSymbols;', 'SdistToGenreGenreRateLU', 'SdistToGenreGenreRateR', 'SdistToGenreGenreRateKJ', 'SdistToGenreGenreRateS', 'SdistToGenreGenreRateGB', 'SdistToGenreGenreRateGE', 'SdistToGenreGenreRateK', 'SdistToGenreGenreRateAG', 'GBdistToGenreWordsPerSentence', 'GBdistToGenreNumberSentences', 'GBdistToGenrePercentageNouns', 'GBdistToGenrePercentageVerbs', 'GBdistToGenrePercentageAdjectives', 'GBdistToGenreNumberCommas', 'GBdistToGenreNumberSymbols€', 'GBdistToGenreNumberSymbolsH', 'GBdistToGenreNumberSymbolsD', 'GBdistToGenreNumberSymbols%', 'GBdistToGenreNumberSymbols§', 'GBdistToGenreNumberSymbols&', 'GBdistToGenreNumberSymbols*', 'GBdistToGenreNumberSymbolsQ', 'GBdistToGenreNumberSymbols-', 'GBdistToGenreNumberSymbols:', 'GBdistToGenreNumberSymbols;', 'GBdistToGenreGenreRateLU', 'GBdistToGenreGenreRateR', 'GBdistToGenreGenreRateKJ', 'GBdistToGenreGenreRateS', 'GBdistToGenreGenreRateGB', 'GBdistToGenreGenreRateGE', 'GBdistToGenreGenreRateK', 'GBdistToGenreGenreRateAG', 'GEdistToGenreWordsPerSentence', 'GEdistToGenreNumberSentences', 'GEdistToGenrePercentageNouns', 'GEdistToGenrePercentageVerbs', 'GEdistToGenrePercentageAdjectives', 'GEdistToGenreNumberCommas', 'GEdistToGenreNumberSymbols€', 'GEdistToGenreNumberSymbolsH', 'GEdistToGenreNumberSymbolsD', 'GEdistToGenreNumberSymbols%', 'GEdistToGenreNumberSymbols§', 'GEdistToGenreNumberSymbols&', 'GEdistToGenreNumberSymbols*', 'GEdistToGenreNumberSymbolsQ', 'GEdistToGenreNumberSymbols-', 'GEdistToGenreNumberSymbols:', 'GEdistToGenreNumberSymbols;', 'GEdistToGenreGenreRateLU', 'GEdistToGenreGenreRateR', 'GEdistToGenreGenreRateKJ', 'GEdistToGenreGenreRateS', 'GEdistToGenreGenreRateGB', 'GEdistToGenreGenreRateGE', 'GEdistToGenreGenreRateK', 'GEdistToGenreGenreRateAG', 'KdistToGenreWordsPerSentence', 'KdistToGenreNumberSentences', 'KdistToGenrePercentageNouns', 'KdistToGenrePercentageVerbs', 'KdistToGenrePercentageAdjectives', 'KdistToGenreNumberCommas', 'KdistToGenreNumberSymbols€', 'KdistToGenreNumberSymbolsH', 'KdistToGenreNumberSymbolsD', 'KdistToGenreNumberSymbols%', 'KdistToGenreNumberSymbols§', 'KdistToGenreNumberSymbols&', 'KdistToGenreNumberSymbols*', 'KdistToGenreNumberSymbolsQ', 'KdistToGenreNumberSymbols-', 'KdistToGenreNumberSymbols:', 'KdistToGenreNumberSymbols;', 'KdistToGenreGenreRateLU', 'KdistToGenreGenreRateR', 'KdistToGenreGenreRateKJ', 'KdistToGenreGenreRateS', 'KdistToGenreGenreRateGB', 'KdistToGenreGenreRateGE', 'KdistToGenreGenreRateK', 'KdistToGenreGenreRateAG', 'AGdistToGenreWordsPerSentence', 'AGdistToGenreNumberSentences', 'AGdistToGenrePercentageNouns', 'AGdistToGenrePercentageVerbs', 'AGdistToGenrePercentageAdjectives', 'AGdistToGenreNumberCommas', 'AGdistToGenreNumberSymbols€', 'AGdistToGenreNumberSymbolsH', 'AGdistToGenreNumberSymbolsD', 'AGdistToGenreNumberSymbols%', 'AGdistToGenreNumberSymbols§', 'AGdistToGenreNumberSymbols&', 'AGdistToGenreNumberSymbols*', 'AGdistToGenreNumberSymbolsQ', 'AGdistToGenreNumberSymbols-', 'AGdistToGenreNumberSymbols:', 'AGdistToGenreNumberSymbols;', 'AGdistToGenreGenreRateLU', 'AGdistToGenreGenreRateR', 'AGdistToGenreGenreRateKJ', 'AGdistToGenreGenreRateS', 'AGdistToGenreGenreRateGB', 'AGdistToGenreGenreRateGE', 'AGdistToGenreGenreRateK', 'AGdistToGenreGenreRateAG', 'minDistToGenreWordsPerSentence', 'minDistToGenreNumberSentences', 'minDistToGenrePercentageNouns', 'minDistToGenrePercentageVerbs', 'minDistToGenrePercentageAdjectives', 'minDistToGenreNumberCommas', 'minDistToGenreNumberSymbols€', 'minDistToGenreNumberSymbolsH', 'minDistToGenreNumberSymbolsD', 'minDistToGenreNumberSymbols%', 'minDistToGenreNumberSymbols§', 'minDistToGenreNumberSymbols&', 'minDistToGenreNumberSymbols*', 'minDistToGenreNumberSymbolsQ', 'minDistToGenreNumberSymbols-', 'minDistToGenreNumberSymbols:', 'minDistToGenreNumberSymbols;', 'minDistToGenreGenreRateLU', 'minDistToGenreGenreRateR', 'minDistToGenreGenreRateKJ', 'minDistToGenreGenreRateS', 'minDistToGenreGenreRateGB', 'minDistToGenreGenreRateGE', 'minDistToGenreGenreRateK', 'minDistToGenreGenreRateAG', 'AuthorGenreRateLU', 'AuthorGenreRateR', 'AuthorGenreRateKJ', 'AuthorGenreRateS', 'AuthorGenreRateGB', 'AuthorGenreRateGE', 'AuthorGenreRateK', 'AuthorGenreRateAG']]
    y_test=tdataFrame['Genre']

    with open("verboseTrainDataFrame.txt", "w") as file:
        file.write(str(dataFrame.head(10)))

def multiOutPrep(outLine, threshhold, step):
    lineOutput = []
    outLine = list(outLine)
    for j in outLine:
        if outLine.index(j) == 0 and j >= threshhold:
            #lineOutput.append('Literatur & Unterhaltung')
            lineOutput.append('Architektur & Garten')
        if outLine.index(j) == 1 and j >= threshhold:
            #lineOutput.append('Ratgeber')
            lineOutput.append('Ganzheitliches Bewusstsein')
        if outLine.index(j) == 2 and j >= threshhold:
            #lineOutput.append('Kinderbuch & Jugendbuch')
            lineOutput.append('Glaube & Ethik')
        if outLine.index(j) == 3 and j >= threshhold:
            #lineOutput.append('Sachbuch')
            lineOutput.append('Kinderbuch & Jugendbuch')
        if outLine.index(j) == 4 and j >= threshhold:
            #lineOutput.append('Ganzheitliches Bewusstsein')
            lineOutput.append('Künste')
        if outLine.index(j) == 5 and j >= threshhold:
            #lineOutput.append('Glaube & Ethik')
            lineOutput.append('Literatur & Unterhaltung')
        if outLine.index(j) == 6 and j >= threshhold:
            #lineOutput.append('Künste')
            lineOutput.append('Ratgeber')
        if outLine.index(j) == 7 and j >= threshhold:
            #lineOutput.append('Architektur & Garten')
            lineOutput.append('Sachbuch')
    if lineOutput == []:
        newthresh = (threshhold - step)
        lineOutput = multiOutPrep(outLine, newthresh, step)
    return lineOutput

def multiOutPrep2(outLine, threshhold):
    lineOutput = []
    outLine = list(outLine)
    maxT = max(outLine)
    for j in outLine:
        if outLine.index(j) == 0 and j >= (maxT - threshhold):
            lineOutput.append('Architektur & Garten')
        if outLine.index(j) == 1 and j >= (maxT - threshhold):
            lineOutput.append('Ganzheitliches Bewusstsein')
        if outLine.index(j) == 2 and j >= (maxT - threshhold):
            lineOutput.append('Glaube & Ethik')
        if outLine.index(j) == 3 and j >= (maxT - threshhold):
            lineOutput.append('Kinderbuch & Jugendbuch')
        if outLine.index(j) == 4 and j >= (maxT - threshhold):
            lineOutput.append('Künste')
        if outLine.index(j) == 5 and j >= (maxT - threshhold):
            lineOutput.append('Literatur & Unterhaltung')
        if outLine.index(j) == 6 and j >= (maxT - threshhold):
            lineOutput.append('Ratgeber')
        if outLine.index(j) == 7 and j >= (maxT - threshhold):
            lineOutput.append('Sachbuch')
    return lineOutput

def multiOutPrep3(outLine, threshhold):
    lineOutput = []
    outLineList = list(outLine)
    maxT = list(outLine)
    maxT.sort(reverse=True)
    currVal = 0
    for j in maxT:
        if currVal < threshhold:
            if outLineList.index(j) == 0:
                lineOutput.append('Architektur & Garten')
            if outLineList.index(j) == 1:
                lineOutput.append('Ganzheitliches Bewusstsein')
            if outLineList.index(j) == 2:
                lineOutput.append('Glaube & Ethik')
            if outLineList.index(j) == 3:
                lineOutput.append('Kinderbuch & Jugendbuch')
            if outLineList.index(j) == 4:
                lineOutput.append('Künste')
            if outLineList.index(j) == 5:
                lineOutput.append('Literatur & Unterhaltung')
            if outLineList.index(j) == 6:
                lineOutput.append('Ratgeber')
            if outLineList.index(j) == 7:
                lineOutput.append('Sachbuch')
            currVal += j
    return lineOutput

def weightTrainData():
    global weights
    global data

    length = len(data)
    """
    luc = 0
    rc = 0
    kjc = 0
    sc = 0
    gbc = 0
    gec = 0
    kc = 0
    agc = 0
    for book in data:
        if book[-1] == "Literatur & Unterhaltung":
            #weights.append("LU")
            luc += 1
        if book[-1] == "Ratgeber":
            #weights.append("R")
            rc += 1
        if book[-1] == "Kinderbuch & Jugendbuch":
            #weights.append("KJ")
            kjc += 1
        if book[-1] == "Sachbuch":
            #weights.append("S")
            sc += 1
        if book[-1] == "Ganzheitliches Bewusstsein":
            #weights.append("GB")
            gbc += 1
        if book[-1] == "Glaube & Ethik":
            #weights.append("GE")
            gec += 1
        if book[-1] == "Künste":
            #weights.append("K")
            kc += 1
        if book[-1] == "Architektur & Garten":
            #weights.append("AG")
            agc += 1
    """

    weights["Literatur & Unterhaltung"] = (length / 8) / length
    weights["Ratgeber"] = (length / 8) / length
    weights["Kinderbuch & Jugendbuch"] = (length / 8) / length
    weights["Sachbuch"] = (length / 8) / length
    weights["Ganzheitliches Bewusstsein"] = (length / 8) / length
    weights["Glaube & Ethik"] = (length / 8) / length
    weights["Künste"] = (length / 8) / length
    weights["Architektur & Garten"] = (length / 8) / length

def trainClassifier():
    global X_train
    global X_test
    global y_test
    global y_train
    global y_predRF
    global weights
    """
    RandomForest Klassifikator trainieren und predicten.
    """

    randomForestClassifier=RandomForestClassifier(n_estimators=100, max_depth=50, class_weight="balanced", min_samples_leaf=1, bootstrap=False, criterion='gini', n_jobs=2)
    randomForestClassifier.fit(X_train, y_train)

    currPos = 0
    if multiOut:
        y_predRF = []
        y_predProb=randomForestClassifier.predict_proba(X_test)
        for i in y_predProb:
            #y_predRF.append(multiOutPrep(i, 0.7, 0.1))
            y_predRF.append(multiOutPrep2(i, 0.1))
            #y_predRF.append(multiOutPrep3(i, 0.6))
            currPos += 1
    else:
        y_predRF=randomForestClassifier.predict(X_test)
        print("F-Score RandomForest:", metrics.f1_score(y_test,  y_predRF, average='micro'))

    """
    gridSearchForest = RandomForestClassifier()
    params = {"n_estimators":[100], "max_depth": [50], "min_samples_leaf":[1], "bootstrap":[False], "criterion":['gini'], "n_jobs":[2], "random_state":[2, 21, 24, 42, 72]}
    clf = GridSearchCV(gridSearchForest, param_grid=params, cv=5)
    clf.fit(X_train, y_train)

    print(clf.best_params_)
    print(clf.best_score_)
    """

    if verbose:
        print("ConfusionMatrix RandomForest:\n", metrics.confusion_matrix(y_test,  y_predRF, labels=["Literatur & Unterhaltung", "Ratgeber", "Kinderbuch & Jugendbuch", "Sachbuch", "Ganzheitliches Bewusstsein", "Glaube & Ethik", "Künste", "Architektur & Garten"]))

def verboseOutput():
    global y_test
    global y_predRF
    """
    Zusaetzliche Valierungsausgabe in Terminal und Datei(OuputData/verboseOutput.txt)
    """
    if verbose:
        prfs = metrics.precision_recall_fscore_support(y_test,  y_predRF,  average='micro')
        print(prfs[0], prfs[1], prfs[2], prfs[3])
        prfsLabel = metrics.precision_recall_fscore_support(y_test,  y_predRF,  average=None, labels=["Literatur & Unterhaltung", "Ratgeber", "Kinderbuch & Jugendbuch", "Sachbuch", "Ganzheitliches Bewusstsein", "Glaube & Ethik", "Künste", "Architektur & Garten"])
        print("LU-precision:", prfsLabel[0][0])
        print("R-precision:", prfsLabel[0][1])
        print("KJ-precision:", prfsLabel[0][2])
        print("S-precision:", prfsLabel[0][3])
        print("GB-precision:", prfsLabel[0][4])
        print("GE-precision:", prfsLabel[0][5])
        print("K-precision:", prfsLabel[0][6])
        print("AG-precision:", prfsLabel[0][7])

        print("LU-recall:", prfsLabel[1][0])
        print("R-recall:", prfsLabel[1][1])
        print("KJ-recall:", prfsLabel[1][2])
        print("S-recall:", prfsLabel[1][3])
        print("GB-recall:", prfsLabel[1][4])
        print("GE-recall:", prfsLabel[1][5])
        print("K-recall:", prfsLabel[1][6])
        print("AG-recall:", prfsLabel[1][7])

        print("LU-fscore:", prfsLabel[2][0])
        print("R-fscore:", prfsLabel[2][1])
        print("KJ-fscore:", prfsLabel[2][2])
        print("S-fscore:", prfsLabel[2][3])
        print("GB-fscore:", prfsLabel[2][4])
        print("GE-fscore:", prfsLabel[2][5])
        print("K-fscore:", prfsLabel[2][6])
        print("AG-fscore:", prfsLabel[2][7])

    """
    with open("OutputData/verboseResults.txt", "w") as file:
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
        file.write("ConfusionMatrix RandomForest:\n" + str(metrics.confusion_matrix(y_test,  y_predRF, labels=["Literatur & Unterhaltung", "Ratgeber", "Kinderbuch & Jugendbuch", "Sachbuch", "Ganzheitliches Bewusstsein", "Glaube & Ethik", "Künste", "Architektur & Garten"])) + str("\n"))
    """

def generateFinalOutputFile():
    global isbnData
    global y_predRF
    """
    Ausgabe Ergebnis-Datei und Baum
    """

    j = 0
    with open("../evaluation/input/answer.txt", "w") as file:
        file.write("subtask_a\n")
        if multiOut:
            for i in isbnData:
                labelStr = ''
                labels = set(y_predRF[j])
                for k in labels:
                    labelStr +=  str("\t") + str(k)
                file.write(i + str(labelStr) + str("\n"))
                j += 1
        else:
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
parser.add_argument("-sig", help="use sigmoid", action="store_true")
parser.add_argument("-gomp", help="use gompertz", action="store_true")
parser.add_argument("-old", help="use old/simple function", action="store_true")
parser.add_argument("-v", help="activate verbose output", action="store_true")
parser.add_argument("-m", help="activate multilabel classification", action="store_true")
parser.add_argument("-x", help="simulation of real use", action="store_true")
parser.add_argument("-rx", help="simulation of real use recursivly", action="store_true")
parser.add_argument("-x10", help="10 cross val with all train data", action="store_true")
parser.add_argument("-val", help="validation", action="store_true")
parser.add_argument("-s", help="enable shuffle", action="store_true")
parser.add_argument("-mp", help="activate multi processing", action="store_true")
parser.add_argument("-mo", help="activate multi output", action="store_true")
args = parser.parse_args()
if args.lv:
    lemmatizeVerbs = True
if args.ln:
    lemmatizeNouns = True
if args.sig:
    useSigmoid = True
if args.gomp:
    useGomp = True
if args.old:
    useOldRec = True
if args.la:
    lemmatizeAdjectives = True
if args.v:
    verbose = True
if args.m:
    multilabel = True
if args.s:
    shuffle = True
if args.mp:
    multiprocessing = True
if args.mo:
    multiOut = True
if args.x10:
    start = timeit.default_timer()
    bookArray = readData('Data/blurbs_train_all.txt')
    stopWordListRead()
    print("Done reading files.",  timeit.default_timer() - start)
    createTempDict()
    improveDict()
    print("Done creating dictionary.",  timeit.default_timer() - start)
    createTrainDataArray()
    meanLU, meanR, meanKJ, meanS, meanGB, meanGE, meanK, meanAG = meanFeatureAll()
    for row in data:
        meanFeatures(row)
    createDataFrames()
    print("Done creating DataFrame.",  timeit.default_timer() - start)
    randomForestClassifier=RandomForestClassifier(n_estimators=100, max_depth=50, min_samples_leaf=1, bootstrap=False, criterion='gini', n_jobs=2)
    crossVal = cross_val_score(randomForestClassifier, X_train, y_train, cv=10, scoring='f1_micro')
    print(crossVal)
    print("10-Cross:", np.mean(crossVal))
    stop = timeit.default_timer()
    print("Runntime: ",  stop - start)
if args.x:
    start = timeit.default_timer()
    bookArray = readData('Data/blurbs_train_all.txt')
    stopWordListRead()
    splitData()
    print("Done reading files.",  timeit.default_timer() - start)
    createTempDict()
    improveDict()
    print("Done creating dictionary.",  timeit.default_timer() - start)
    createTrainDataArray()
    createTestDataArray()
    weightTrainData()
    meanLU, meanR, meanKJ, meanS, meanGB, meanGE, meanK, meanAG = meanFeatureAll()
    for row in data:
        meanFeatures(row)
    for row in tdata:
        meanFeatures(row)
    a = 0
    for row in data:
        for i in authorRatesTrain[a]:
            row.append(i)
        for j in biGenreRatesTrain[a]:
            row.append(j)
        a += 1
    a = 0
    for row in tdata:
        for i in authorRatesTrain[a]:
            row.append(i)
        for j in biGenreRatesTrain[a]:
            row.append(j)
        a += 1
    createDataFrames()
    print("Done creating DataFrame.",  timeit.default_timer() - start)
    trainClassifier()
    print("Min LU: ",  min(dictLU.items(),  key=lambda x: x[1]))
    print("Max LU: ",  max(dictLU.items(),  key=lambda x: x[1]))
    #print("MinBi LU: ",  min(biDictLU.items(),  key=lambda x: x[1]))
    #print("MaxBi LU: ",  max(biDictLU.items(),  key=lambda x: x[1]))
    #print("MinTri LU: ",  min(triDictLU.items(),  key=lambda x: x[1]))
    #print("MaxTri LU: ",  max(triDictLU.items(),  key=lambda x: x[1]))
    #print("MinQuadro LU: ",  min(quadroDictLU.items(),  key=lambda x: x[1]))
    #print("MaxQuadro LU: ",  max(quadroDictLU.items(),  key=lambda x: x[1]))
    #print("MinPenta LU: ",  min(pentaDictLU.items(),  key=lambda x: x[1]))
    #print("MaxPenta LU: ",  max(pentaDictLU.items(),  key=lambda x: x[1]))
    #print("MinAu LU: ",  min(auDictLU.items(),  key=lambda x: x[1]))
    #print("MaxAu LU: ",  max(auDictLU.items(),  key=lambda x: x[1]))
    generateFinalOutputFile()
    stop = timeit.default_timer()
    print("Runntime: ",  stop - start)
if args.val:
    start = timeit.default_timer()
    bookArray = readData('Data/blurbs_train_all.txt')
    tbookArray = readData('Data/blurbs_test_participant.txt')
    stopWordListRead()
    print("Done reading files.",  timeit.default_timer() - start)
    createTempDict()
    improveDict()
    print("Done creating dictionary.",  timeit.default_timer() - start)
    createTrainDataArray()
    createTestDataArray()
    meanLU, meanR, meanKJ, meanS, meanGB, meanGE, meanK, meanAG = meanFeatureAll()
    for row in data:
        meanFeatures(row)
    for row in tdata:
        meanFeatures(row)
    createDataFrames()
    print("Done creating DataFrame.",  timeit.default_timer() - start)
    trainClassifier()
    generateFinalOutputFile()
    stop = timeit.default_timer()
    print("Runntime: ",  stop - start)
if args.rx:
    start = timeit.default_timer()
    fscores =[]
    bookArray = readData('Data/blurbs_train_all.txt')
    stopWordListRead()
    splitData()
    tbookArrayRec = splitter(tbookArray, 50)
    for part in tbookArrayRec:
        tbookArray = part
        print("Done reading files.",  timeit.default_timer() - start)
        createTempDict()
        improveDict()
        print("Done creating dictionary.",  timeit.default_timer() - start)
        createTrainDataArray()
        createTestDataArray()
        meanLU, meanR, meanKJ, meanS, meanGB, meanGE, meanK, meanAG = meanFeatureAll()
        for row in data:
            meanFeatures(row)
        for row in tdata:
            meanFeatures(row)
        createDataFrames()
        print("Done creating DataFrame.",  timeit.default_timer() - start)
        trainClassifier()
        recCounter = 0
        for i in part:
            i = list(i)
            i[5] = y_predRF[recCounter]
            recCounter += 1
            i = tuple(i)
            bookArray.append(i)
        fscores.append(metrics.f1_score(y_test,  y_predRF, average='micro'))
        curr = 0
    print("All F-Scores: ",  fscores)
    print("Recursive F-Score: ",  np.mean(fscores))
    stop = timeit.default_timer()
    print("Runntime: ",  stop - start)
