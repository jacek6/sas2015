from os import path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics, cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import *
import re
import numpy as np
import matplotlib.pyplot as plt

notebook_path = 'C:/Users/Ola1/Desktop/'
data = pd.read_csv(path.join(notebook_path, 'SemEval-2014.csv'), index_col=0)
docs = data['document'] 
y = data['sentiment'] # standart name for labels/classes variable 
count_vect = CountVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english')
X = count_vect.fit_transform(docs)


fp = open("C:/Users/Ola1/Desktop/tekst.txt")
data = fp.read()
data = data.replace("\n", " ")

allData = re.split("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", data)
cameraData = []
batteryData = []
screenData = []
cpuData = []

for sentence in allData:
    if "camera".upper() in sentence.upper() or "photo".upper() in sentence.upper():
        cameraData.append(sentence)
    if "battery".upper() in sentence.upper():
        batteryData.append(sentence)
    if "screen".upper() in sentence.upper() or "resolution".upper() in sentence.upper() or "display".upper() in sentence.upper() or "colour".upper() in sentence.upper() or "color".upper() in sentence.upper():
        screenData.append(sentence)
    if "cpu".upper() in sentence.upper() or "processor".upper() in sentence.upper() or "speed".upper() in sentence.upper():
        cpuData.append(sentence)

newVec = CountVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english', vocabulary=count_vect.vocabulary_)        
allDataX = newVec.fit_transform(allData)
cameraDataX = newVec.fit_transform(cameraData)
batteryDataX = newVec.fit_transform(batteryData)
screenDataX = newVec.fit_transform(screenData)
cpuDataX = newVec.fit_transform(cpuData)
        
def sentiment_classification(X, y, n_folds=10, classifier=None):
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
        predicted = clf.predict(screenDataX)
        for i in range(0, len(predicted)-1):
            print (screenData[i], predicted[i], "\n")
        scMean = (np.mean(predicted)-1)*50
        
        predicted = clf.predict(cameraDataX)
        cmMean = (np.mean(predicted)-1)*50
        predicted = clf.predict(batteryDataX)
        btMean = (np.mean(predicted)-1)*50
        predicted = clf.predict(cpuDataX)
        cpuMean = (np.mean(predicted)-1)*50
        predicted = clf.predict(allDataX)
        allMean = (np.mean(predicted)-1)*50
        
        x = np.array([0,1,2,3,4])
        y = np.array([cmMean, scMean, btMean, cpuMean, allMean])
        my_xticks = ["camera","screen","battery","processor", "all"]
        plt.xticks(x, my_xticks)
        plt.bar(x, y, 0.35)
        plt.ylabel('procent')
        plt.xlabel('kategoria')
        plt.show()
        
        return results


results = sentiment_classification(X, y, n_folds=4, classifier=RidgeClassifier())
#print ('Accuracy: %s, F1-measure: %s, Predicted: %s' % (np.mean(results['acc']), np.mean(results['f1']), np.mean(results['prec'])))

