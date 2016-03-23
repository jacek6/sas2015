#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys, inspect
import pprint

import pandas
from sklearn import metrics, cross_validation
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer

class ClsTest:
    """
    Different classifiers efficiency tester.
    """
    def vectorize(self, docs, ngramRange=(1,2)):
        """
        Convert a collection of text documents to a matrix of token counts.
        Next learn the vocabulary dictionary and return term-document matrix.
        """
        v = CountVectorizer( ngram_range=(1,2), lowercase=True, stop_words='english' )
        return v.fit_transform( docs )
    ##
    def classifySentiment(self, X, y, n_folds=10, classifier=None):
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
    ##
##

if __name__ == "__main__":
    # For notebook:
    #cwd = os.getcwd()
    # For file:
    cwd = os.path.dirname(os.path.realpath(__file__))
    data = pandas.read_csv(os.path.join(cwd, 'data', 'SemEval-2014.csv'), index_col=0)
    docs = data['document'] 
    y = data['sentiment']
    
    clsTest = ClsTest()

    X = clsTest.vectorize( docs, (1,2) )
    
    classifiers = [
        "LogisticRegression",
        "PassiveAggressiveClassifier",
        "Perceptron",
        "RidgeClassifier",
        "SGDClassifier"
    ]
    for cls in classifiers:
        cls = "linear_model." + cls
        results = clsTest.classifySentiment( X=X, y=y, n_folds=4, classifier=eval(cls)() )
        print( "Classifier: '%s' - Result[acc] = %s" % (cls.replace("linear_model.",""), str(results['acc'][0]) ) )
##
