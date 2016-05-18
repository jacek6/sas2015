import warnings
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn import metrics, cross_validation

class CoreLearning:

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
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            ######################## Most important part ##########################
            clf = classifier.fit(X_train, y_train)  # train the classifier
            predicted = clf.predict(X_test)  # predict test the classifier
            #######################################################################

            results['acc'].append(metrics.accuracy_score(y_test, predicted))
            results['prec'].append(metrics.precision_score(y_test, predicted, average='weighted'))
            results['rec'].append(metrics.recall_score(y_test, predicted, average='weighted'))
            results['f1'].append(metrics.f1_score(y_test, predicted, average='weighted'))
            results['cm'].append(metrics.confusion_matrix(y_test, predicted))

        # print ('Accuracy: %s, F1-measure: %s, Predicted: %s' % (np.mean(results['acc']), np.mean(results['f1']), np.mean(results['prec'])))
        metricss = [np.mean(results['acc']), np.mean(results['f1']), np.mean(results['prec'])]
        return clf, metricss

