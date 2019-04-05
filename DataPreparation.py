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

print(bookArray)