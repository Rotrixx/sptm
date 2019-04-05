from textblob_de import TextBlobDE

bodyStr = ''
authorStr = ''
categoryStr = ''
allCategoryStr = set()
isbnStr = ''
bookArray = []

with open('Data/train_small.txt','r') as file:
	for line in file:
		if line.startswith('<book'):
			bodyStr = ''
			authorStr = ''
			categoryStr = ''
			allCategoryStr = set()
			isbnStr = ''
		elif line.startswith('</book>'):
			bookArray.append((bodyStr,authorStr,allCategoryStr,isbnStr))
		elif line.startswith('<body>'):
			bodyStr += line
			bodyStr = bodyStr[:-8]
			bodyStr = bodyStr[6:]
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

print(bookArray[0][0])
print(bookArray[0][1])
print(bookArray[0][2])
print(bookArray[0][3])

def wordsPerSentenceAndNumberSentences(text):
	j = 0
	k = 0
	blob = TextBlobDE(text)
	for sentence in blob.sentences:
		k += 1
		for word in sentence.words:
			j += 1
	j = j / k
	return j,k

def percentageOfNounsVerbsAdjectivesAndNumberCommasSymbols(text):
	nNouns = 0
	nVerbs = 0
	nAdjectives = 0
	nCommas = 0
	nSym = 0
	j = 0
	blob = TextBlobDE(text)
	textTockens = blob.tags
	for i in textTockens:
		j += 1
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

	rNouns = nNouns / j
	rVerbs = nVerbs / j
	rAdjectives = nAdjectives / j
	return rNouns,rVerbs,rAdjectives,nCommas,nSym

print(percentageOfNounsVerbsAdjectivesAndNumberCommasSymbols(bookArray[0][0]))