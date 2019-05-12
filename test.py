from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import json
import random
import numpy as np

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
data2 = []
isbnData = []
currPos = 0

stopwords = None

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
tdata2 = []
tisbnData = []

genreTest = None
genreTrain = None

def readDataOneFile():
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
    # Read TrainData
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
                            for i in allCategoryStr:
                                if bodyStr == '':
                                    continue
                                if isbnStr.startswith('4'):
                                    continue
                                bookArray.append((bodyStr,titleStr,authorStr,allCategoryStr,isbnStr,i))
                            """
                            if bodyStr == '':
                                    continue
                            if isbnStr.startswith('4'):
                                    continue
                            bookArray.append((bodyStr,titleStr,authorStr,allCategoryStr,isbnStr,firstCategory))
                            """
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

def splitter(array, size):
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
    random.shuffle(bookArray)
    helper = splitter(bookArray,10000)
    tbookArray = helper[1]
    bookArray = helper[0]
    print(len(tbookArray))
    print(len(bookArray))

def stopWordListRead():
    global stopwords
    with open('Data/stopwords_german.txt','r') as file:
        stopwords = json.load(file)

def createDataArray():
    global currPos
    global bookArray
    global tbookArray
    global data
    global tdata
    global data2
    global tdata2
    global isbnData
    global tisbnData
    # Creation of TrainDataFrame
    for _ in bookArray:
            data.append(bookArray[currPos][0])
            data2.append(bookArray[currPos][5])
            currPos += 1
    # Creation of TestDataFrame
    currPos = 0
    for _ in tbookArray:
            tdata.append(tbookArray[currPos][0])
            tdata2.append(tbookArray[currPos][5])
            currPos += 1


readDataOneFile()
stopWordListRead()
splitData()
createDataArray()

y_train, y_test = data2, tdata2

vectorizer = HashingVectorizer(stop_words=stopwords, alternate_sign=False,analyzer='word',ngram_range=(1,2),strip_accents='unicode',norm='l2')
X_train = vectorizer.transform(data)
X_test = vectorizer.transform(tdata)

#X_trainTest = vectorizer.transform(data).data[0]
#X_testTest = vectorizer.transform(tdata)

#print(X_testTest)
#print(X_testTest.shape[0])

X_testTest = []
X_trainTest = []

i = 0
traintest = vectorizer.transform(data)

while i < len(data):
    X_trainTest.append(traintest[i].data[0])
    i += 1


i = 0
testtest = vectorizer.transform(tdata)
while i < len(tdata):
    X_testTest.append(testtest[i].data[0])
    i += 1


randomForestClassifier=RandomForestClassifier(n_estimators=10,random_state=0,verbose=10,n_jobs=2)
randomForestClassifier.fit(X_train,y_train)

y_predRF = randomForestClassifier.predict(X_test)

X_trainTest = np.asarray(X_trainTest).reshape(-1,1)
X_testTest = np.asarray(X_testTest).reshape(-1,1)

randomForestClassifierTest=RandomForestClassifier(n_estimators=10,random_state=0,verbose=10,n_jobs=2)
randomForestClassifierTest.fit(X_trainTest,y_train)

y_predRFTest = randomForestClassifier.predict(X_testTest)

print("F-Score RandomForest:",metrics.f1_score(y_test, y_predRF,average='micro'))
print("F-Score RandomForestTest:",metrics.f1_score(y_test, y_predRFTest,average='micro'))
print("ConfusionMatrix RandomForest:\n",metrics.confusion_matrix(y_test, y_predRF,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"]))
print("ConfusionMatrix RandomForestTest:\n",metrics.confusion_matrix(y_test, y_predRFTest,labels=["Literatur & Unterhaltung","Ratgeber","Kinderbuch & Jugendbuch","Sachbuch","Ganzheitliches Bewusstsein","Glaube & Ethik","Künste","Architektur & Garten"]))
