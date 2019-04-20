#!/usr/bin/python3.7

from textblob_de import TextBlobDE
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn import metrics

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

# Ensemble
ensembleArray = [[],[],[],[],[]]

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

print('Reading files Done')

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
        
        return j,k,rNouns,rVerbs,rAdjectives,nCommas,nSym,gdrLU,gdrR,gdrKJ,gdrS,gdrGB,gdrGE,gdrK,gdrAG

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

print('Featurize Done')

# TrainDataFrame
dataFrame=pd.DataFrame(data,columns=['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG','Genre'],index=isbnData,dtype=float)

# TestDataFrame
tdataFrame=pd.DataFrame(tdata,columns=['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG','Genre'],dtype=float)

# TrainData
X_train=dataFrame[['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG']]
y_train=dataFrame['Genre']

# TestData
X_test=tdataFrame[['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG']]
y_test=tdataFrame['Genre']

print('DataFrames Done')

# RandomForestClassifier
randomForestClassifier=RandomForestClassifier(n_estimators=50,oob_score=True,random_state=0,class_weight="balanced")
randomForestClassifier.fit(X_train,y_train)

# RadiusNeighborClassifier
radiusNeighborClassifier=RadiusNeighborsClassifier(radius=9.0,outlier_label='Literatur & Unterhaltung')
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

print('All training  Done')

y_predRF=randomForestClassifier.predict(X_test)
y_predRN=radiusNeighborClassifier.predict(X_test)
y_predGNB=gaussianNaiveBayesClassifier.predict(X_test)
y_predMNB=multinominalNaiveBayesClassifier.predict(X_test)
y_predBNB=bernoulliNaiveBayesClassifier.predict(X_test)

print('Predictions Done')

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
ensembleDecision = []
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

#X_ensembleTreeDataFrameTrain = pd.DataFrame(data={'RF' : ensembleArray[0], 'RN' : ensembleArray[1], 'GNB' : ensembleArray[2], 'MNB' : ensembleArray[3], 'BNB' : ensembleArray[4]})

#decisionTree = tree.DecisionTreeClassifier()
#decisionTree.fit(X_ensembleTreeDataFrameTrain,X_test)

#eDT = decisionTree.predict(y_test)

# Result without isbn
# print(ensembleDecision)

print("Accuracy RandomForest:",metrics.accuracy_score(y_test, y_predRF))
print("Accuracy RadiusNeighbor:",metrics.accuracy_score(y_test, y_predRN))
print("Accuracy GaussianNaiveBayes:",metrics.accuracy_score(y_test, y_predGNB))
print("Accuracy MultinominalNaiveBayes:",metrics.accuracy_score(y_test, y_predMNB))
print("Accuracy BernoulliNaiveBayes:",metrics.accuracy_score(y_test, y_predBNB))

print("Accuracy Ensemble:",metrics.accuracy_score(y_test, ensembleDecision))

print("F-Score RandomForest:",metrics.f1_score(y_test, y_predRF,average='micro'))
print("F-Score RadiusNeighbor:",metrics.f1_score(y_test, y_predRN,average='micro'))
print("F-Score GaussianNaiveBayes:",metrics.f1_score(y_test, y_predGNB,average='micro'))
print("F-Score MultinominalNaiveBayes:",metrics.f1_score(y_test, y_predMNB,average='micro'))
print("F-Score BernoulliNaiveBayes:",metrics.f1_score(y_test, y_predBNB,average='micro'))

print("F-Score Ensemble:",metrics.f1_score(y_test, ensembleDecision,average='micro'))
#print("F-Score EnsembleTree:",metrics.f1_score(y_test, eDT,average='micro'))


"""
Accuracy RandomForest: 0.728448275862069
Accuracy NearestCentroid: 0.09913793103448276
Accuracy KNearestNeighbor: 0.5086206896551724
Accuracy RadiusNeighbor: 0.5387931034482759
Accuracy NaiveBayes: 0.21551724137931033
Accuracy GaussianNaiveBayes: 0.7327586206896551
Accuracy MultinominalNaiveBayes: 0.5431034482758621
Accuracy BernoulliNaiveBayes: 0.5387931034482759
Accuracy PassiveAggressive: 0.4353448275862069
"""
