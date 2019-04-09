#!/usr/bin/python3.7

from textblob_de import TextBlobDE
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

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

print(bookArray[0][0])
print(bookArray[0][1])
print(bookArray[0][2])
print(bookArray[0][3])
print(bookArray[0][4])

def featurize(text):
	j = 0
	k = 0
	nNouns = 0
	nVerbs = 0
	nAdjectives = 0
	nCommas = 0
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
		elif i[1] == ', ':
			nCommas += 1
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

	return j,k,rNouns,rVerbs,rAdjectives,nCommas,nSym,gdrLU,gdrR,gdrKJ,gdrS,gdrGB,gdrGE,gdrK,gdrAG

for i in bookArray:
	wps,ns,rn,rv,ra,nc,nsym,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG = featurize(bookArray[currPos][0])
	data.append([wps,ns,rn,rv,ra,nc,nsym,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG,bookArray[currPos][5]])
	isbnData.append(bookArray[currPos][4])
	currPos += 1

dataFrame=pd.DataFrame(data,columns=['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG','Genre'],index=isbnData,dtype=float)

print(dataFrame)

X=dataFrame[['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG']]  # Features
y=dataFrame['Genre']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=50,oob_score=True,random_state=0,class_weight="balanced")

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))