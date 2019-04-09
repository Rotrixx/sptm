#!/usr/bin/python3.7

from textblob_de import TextBlobDE
import pandas as pd
import json

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

bodyStr = ''
titleStr = ''
authorStr = ''
categoryStr = ''
allCategoryStr = set()
isbnStr = ''
bookArray = []
data = []
isbnData = []
currPos = 0

with open('Data/train_small.txt','r') as file:
	for line in file:
		if line.startswith('<book'):
			bodyStr = ''
			titleStr = ''
			authorStr = ''
			categoryStr = ''
			allCategoryStr = set()
			isbnStr = ''
		elif line.startswith('</book>'):
			bookArray.append((bodyStr,titleStr,authorStr,allCategoryStr,isbnStr))
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

	gdrLU = grLU / (grLU+grAG+grK+grGE+grGB+grS+grKJ+grR)
	gdrR = grR / (grLU+grAG+grK+grGE+grGB+grS+grKJ+grR)
	gdrKJ = grKJ / (grLU+grAG+grK+grGE+grGB+grS+grKJ+grR)
	gdrS = grS / (grLU+grAG+grK+grGE+grGB+grS+grKJ+grR)
	gdrGB = grGB / (grLU+grAG+grK+grGE+grGB+grS+grKJ+grR)
	gdrGE = grGE / (grLU+grAG+grK+grGE+grGB+grS+grKJ+grR)
	gdrK = grK / (grLU+grAG+grK+grGE+grGB+grS+grKJ+grR)
	gdrAG = grAG / (grLU+grAG+grK+grGE+grGB+grS+grKJ+grR)

	return j,k,rNouns,rVerbs,rAdjectives,nCommas,nSym,gdrLU,gdrR,gdrKJ,gdrS,gdrGB,gdrGE,gdrK,gdrAG

for i in bookArray:
	wps,ns,rn,rv,ra,nc,nsym,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG = featurize(bookArray[currPos][0])
	data.append([wps,ns,rn,rv,ra,nc,nsym,rLU,rR,rKJ,rS,rGB,rGE,rK,rAG])
	isbnData.append(bookArray[currPos][4])
	currPos += 1

dataFrame=pd.DataFrame(data,columns=['WordsPerSentence','NumberSentences','PercentageNouns','PercentageVerbs','PercentageAdjectives','NumberCommas','NumberSymbols','GenreRateLU','GenreRateR','GenreRateKJ','GenreRateS','GenreRateGB','GenreRateGE','GenreRateK','GenreRateAG'],index=isbnData,dtype=float)

print(dataFrame)