from base1 import *

class MyCls:
    def __init__(self):
        pass

    def fit(self, X, y):
        print u"X[0] size = {0:d}".format(X[0].size)
        xx = X[0] + X[1]
        print "xx.size = %s" % (dir(xx))
        self.yy = y
        return self

    def predict(self, X):
        result = np.zeros(shape=(X.shape[0]))
        for i in range(0, result.size - 1):
            result[i] = 1
        return result

#results = sentiment_classification(X, y, n_folds=4, classifier=LogisticRegression(), printing=False)
results = sentiment_classification(X, y, n_folds=4, classifier=MyCls(), printing=False)
print 'Accuracy: %s' % np.mean(results['acc'])
print 'F1-measure: %s' % np.mean(results['f1'])
