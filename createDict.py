from textblob_de import TextBlobDE
import json

bodyStr = ''
bookArray = []
categoryStr = ''
allCategoryStr = set()
curr = 0

dictLU = {}
dictR = {}
dictKJ = {}
dictS = {}
dictGB = {}
dictGE = {}
dictK = {}
dictAG = {}

def addToDict(word):
	global curr
	if 'Literatur & Unterhaltung' in bookArray[curr][1]:
		dictLU[word] = 1
	if 'Ratgeber' in bookArray[curr][1]:
		dictR[word] = 1
	if 'Kinderbuch & Jugendbuch' in bookArray[curr][1]:
		dictKJ[word] = 1
	if 'Sachbuch' in bookArray[curr][1]:
		dictS[word] = 1
	if 'Ganzheitliches Bewusstsein' in bookArray[curr][1]:
		dictGB[word] = 1
	if 'Glaube & Ethik' in bookArray[curr][1]:
		dictGE[word] = 1
	if 'Künste' in bookArray[curr][1]:
		dictK[word] = 1
	if 'Architektur & Garten' in bookArray[curr][1]:
		dictAG[word] = 1

with open('Data/blurbs_train2.txt','r') as file:
	for line in file:
		if line.startswith('<book'):
			bodyStr = ''
			categoryStr = ''
			allCategoryStr = set()
		elif line.startswith('</book>'):
			bookArray.append((bodyStr,allCategoryStr))
		elif line.startswith('<body>'):
			bodyStr += line
			bodyStr = bodyStr[:-8]
			bodyStr = bodyStr[6:]
		elif line.startswith('<topic d="0">'):
			categoryStr += line
			categoryStr = categoryStr[:-9]
			categoryStr = categoryStr[13:]
			allCategoryStr.add(categoryStr)
			categoryStr = ''

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

with open('dictLU.txt','w') as file:
	file.write(json.dumps(dictLU))
with open('dictR.txt','w') as file:
	file.write(json.dumps(dictR))
with open('dictKJ.txt','w') as file:
	file.write(json.dumps(dictKJ))
with open('dictS.txt','w') as file:
	file.write(json.dumps(dictS))
with open('dictGB.txt','w') as file:
	file.write(json.dumps(dictGB))
with open('dictGE.txt','w') as file:
	file.write(json.dumps(dictGE))
with open('dictK.txt','w') as file:
	file.write(json.dumps(dictK))
with open('dictAG.txt','w') as file:
	file.write(json.dumps(dictAG))
