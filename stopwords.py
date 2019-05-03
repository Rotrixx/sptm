from nltk.corpus import stopwords
import json


stopWords = set(stopwords.words('german'))
# print(stopWords)

with open('stopwords_german.txt','w') as file:
        file.write(str(stopWords))

