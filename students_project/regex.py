from os import path
notebook_path = 'C:/Users/Ola1/Desktop/'
data = pd.read_csv(path.join(notebook_path, 'SemEval-2014.csv'), index_col=0)
import pandas as pd
docs = data['document'] 
y = data['sentiment'] # standart name for labels/classes variable 
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english')
X = count_vect.fit_transform(docs)
from sklearn import metrics, cross_validation
from sklearn.linear_model import LogisticRegression

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

        return results

from sklearn.linear_model import *
#results = sentiment_classification(X, y, n_folds=4, classifier=RidgeClassifier())
#import numpy as np
#print ('Accuracy: %s, F1-measure: %s' % (np.mean(results['acc']), np.mean(results['f1'])))

'''import nltk.data
tokenizer = nltk.data.load('C:/Users/Ola1/Anaconda3/Lib/site-packages/nltk/tokenize/punkt/english.pickle')
fp = open("C:/Users/Ola1/Desktop/tekst.txt")
data = fp.read()
print ('\n-----\n'.join(tokenizer.tokenize(data)))'''

fp = open("C:/Users/Ola1/Desktop/tekst.txt")
data = fp.read()
data = data.replace("\n", " ")

import re
tekst = re.split("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", data)

print (tekst)
