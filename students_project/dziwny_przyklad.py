import pandas as pd
from os import path
import numpy as np

notebook_path = '../'
data = pd.read_csv(path.join(notebook_path, 'data', 'SemEval-2014.csv'), index_col=0)
docs = data['document']
y = data['sentiment'] # standart name for labels/classes variable
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english')
X = count_vect.fit_transform(docs)
from sklearn import metrics, cross_validation
from sklearn.linear_model import LogisticRegression

def sentiment_classification(X, y, n_folds=10, classifier=None, printing=False):
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
            if printing: print "train_index = %s, test_index %s" % (train_index.size, test_index.size)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            ######################## Most important part ##########################
            if printing: print 'X_train = %s, y_train = %s' % (X_train.shape, y_train.shape)
            clf = classifier.fit(X_train, y_train) # train the classifier
            predicted = clf.predict(X_test) # predict test the classifier
            if printing: print 'X_test = %s, predicted = %s' % (X_test.shape, predicted.shape)
            #######################################################################

            results['acc'].append(metrics.accuracy_score(y_test, predicted))
            results['prec'].append(metrics.precision_score(y_test, predicted, average='weighted'))
            results['rec'].append(metrics.recall_score(y_test, predicted, average='weighted'))
            results['f1'].append(metrics.f1_score(y_test, predicted, average='weighted'))
            results['cm'].append(metrics.confusion_matrix(y_test, predicted))

        return results

# z tym dziala:
results = sentiment_classification(X, y, n_folds=4, classifier=LogisticRegression())
# z tym nie dziala:
results = sentiment_classification(X[1:100], y[1:100], n_folds=4, classifier=LogisticRegression())


print 'Accuracy: %s' % np.mean(results['acc'])
print 'F1-measure: %s' % np.mean(results['f1'])

# CEL: podac metodzie sentiment_classification podzbior X i y