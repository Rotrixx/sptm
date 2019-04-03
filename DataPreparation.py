bookStr = ''
bookArray = []

with open('Data/train_small.txt','r') as file:
	for line in file:
		if line.startswith('<book'):
			bookStr = ''
			bookStr += line
		elif line.startswith('</book>'):
			bookStr += line
			bookArray.append(bookStr)
		else:
			bookStr += line

print(bookArray)