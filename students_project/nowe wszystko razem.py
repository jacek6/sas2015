from os import path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics, cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import *
import re
import numpy as np
import matplotlib.pyplot as plt

#singular value decomposition - kompresja
#wykorzystac nowe labelki, poprawic wykres, zrobic raport
#wykorzystac pca principal component analysis, naiwny bajes
'''import nltk.data
nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')'''

notebook_path = 'C:/Users/Ola1/Desktop/'
test_data_path = "C:/Users/Ola1/Desktop/tekst.txt"
amazon_labels_path = 'C:/Users/Ola1/Desktop/sentiment labelled sentences/amazon_cells_labelled.txt'

y = []
X = []
count_vect = CountVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english')

allData = []
cameraData = []
batteryData = []
screenData = []
cpuData = []

allDataX = []
cameraDataX = []
batteryDataX = []
screenDataX = []
cpuDataX = []

def read_twitter_data(notebook_path):
    global X, y
    data = pd.read_csv(path.join(notebook_path, 'SemEval-2014.csv'), index_col=0)
    docs = data['document'] 
    y = data['sentiment'] # standart name for labels/classes variable 
    X = count_vect.fit_transform(docs)

def read_amazon_data(amazon_labels_path):
    global X, y
    docs = []
    with open(amazon_labels_path) as f:
        for line in f:
            line = line.strip()
            sentim = line[len(line)-1]
            try:
                sen = int(sentim)
                if sen < 2 and sen > -1:
                    y.append(sen)
                    line = line[:-1].strip()
                    docs.append(line)
            except:
                pass
    y = np.array(y)
    X = count_vect.fit_transform(docs)
    
def read_test_data(test_data_path):
    global allData, cameraData, batteryData, screenData, cpuData, allDataX, cameraDataX, batteryDataX, screenDataX, cpuDataX
    fp = open(test_data_path)
    data = fp.read()

    #allData = tokenizer.tokenize(data)
    data = data.replace("\n", " ")
    allData = re.split("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s|[.!?](?!\s)(?!\d)(?![g])", data)
    
    for sentence in list(allData):
        if len(sentence) < 10:
            allData.remove(sentence)
        else:       
            if "CAMERA" in sentence.upper() or "PHOTO" in sentence.upper():
                cameraData.append(sentence)
            if "BATTERY" in sentence.upper():
                batteryData.append(sentence)
            if "SCREEN" in sentence.upper() or "RESOLUTION" in sentence.upper() or "DISPLAY" in sentence.upper() or "COLOUR" in sentence.upper() or "COLOR" in sentence.upper() or "UI" in sentence.upper():
                screenData.append(sentence)
            if "CPU" in sentence.upper() or "PROCESSOR" in sentence.upper() or "SPEED" in sentence.upper():
                cpuData.append(sentence)

    newVec = CountVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english', vocabulary=count_vect.vocabulary_)        
    allDataX = newVec.fit_transform(allData)
    cameraDataX = newVec.fit_transform(cameraData)
    batteryDataX = newVec.fit_transform(batteryData)
    screenDataX = newVec.fit_transform(screenData)
    cpuDataX = newVec.fit_transform(cpuData)
    
def sentiment_classification_learning(X, y, n_folds=10, classifier=None):
        """
        Counting sentiment with cross validation - supervised method
        :type X: ndarray feature matrix for classification
        :type y: list or ndarray of classes
        :type n_folds: int # of folds for CV
        :type classifier: classifier which we train and predict sentiment
        :return: measures: accuracy, precision, recall, f1
        """
        results = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'cm': []}
        kf = cross_validation.StratifiedKFold(y, n_folds=n_folds, shuffle=True)
        print (kf)
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            ######################## Most important part ########################## 
            clf = classifier.fit(X_train, y_train) # train the classifier 
            predicted = clf.predict(X_test) # predict test the classifier 
            #######################################################################

            results['acc'].append(metrics.accuracy_score(y_test, predicted))
            results['prec'].append(metrics.precision_score(y_test, predicted, average='weighted'))
            results['rec'].append(metrics.recall_score(y_test, predicted, average='weighted'))
            results['f1'].append(metrics.f1_score(y_test, predicted, average='weighted'))
            results['cm'].append(metrics.confusion_matrix(y_test, predicted))
        
        print ('Accuracy: %s, F1-measure: %s, Predicted: %s' % (np.mean(results['acc']), np.mean(results['f1']), np.mean(results['prec'])))        
        return clf

def sentiment_classification_testing(clf, minus, multiplier):
    predicted = clf.predict(screenDataX)
    scMean = (np.mean(predicted)-minus)*multiplier    
    predicted = clf.predict(cameraDataX)
    cmMean = (np.mean(predicted)-minus)*multiplier
    predicted = clf.predict(batteryDataX)
    btMean = (np.mean(predicted)-minus)*multiplier       
    predicted = clf.predict(cpuDataX)
    cpuMean = (np.mean(predicted)-minus)*multiplier
    predicted = clf.predict(allDataX)
    allMean = (np.mean(predicted)-minus)*multiplier

    for i in range(0, len(predicted)-1):
        print (allData[i], predicted[i], "\n")

    x = np.array([0,1,2,3,4])
    y = np.array([cmMean, scMean, btMean, cpuMean, allMean])
    my_xticks = ["camera","screen","battery","processor", "all"]
    plt.xticks(x, my_xticks)
    plt.bar(x, y, 0.35)
    plt.ylabel('procent')
    plt.xlabel('kategoria')
    plt.show()

#read_twitter_data(notebook_path)
read_amazon_data(amazon_labels_path)
clf = sentiment_classification_learning(X, y, n_folds=4, classifier=LogisticRegression())

read_test_data(test_data_path)
#sentiment_classification_testing(clf, 1, 50)
sentiment_classification_testing(clf, 0, 100)