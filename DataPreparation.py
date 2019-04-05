from textblob_de import TextBlobDE

bookStr = ''
bodyStr = ''
bookArray = []

with open('Data/train_small.txt','r') as file:
	for line in file:
		if line.startswith('<book'):
			bookStr = ''
			bodyStr = ''
			bookStr += line
		elif line.startswith('</book>'):
			bookStr += line
			bookArray.append((bookStr,bodyStr))
		elif line.startswith('<body>'):
			bookStr += line
			bodyStr += line
			bodyStr = bodyStr[:-7]
			bodyStr = bodyStr[6:]
		else:
			bookStr += line

print(bookArray[0][1])

def wordsPerSentenceAndNumberSentences(text):
	j = 0
	k = 0
	blob = TextBlobDE(text)
	for sentence in blob.sentences:
		k += 1 #
		for word in sentence.words:
			j += 1
	j = j / k
	return j,k

print(wordsPerSentenceAndNumberSentences(bookArray[0][1]))