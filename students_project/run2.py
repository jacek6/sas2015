from base1 import *


class MyCls:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.yy = y
        return self

    def predict(self, X):
        result = np.zeros(shape=(X.shape[0]))
        for i in range(0, result.size - 1):
            result[i] = 2
        return result


class ConstantClassifier:
    def __init__(self, con):
        self.c = con

    def fit(self, X, y):
        return self

    def predict(self, X):
        result = np.zeros(shape=(X.shape[0]))
        for i in range(0, result.size - 1):
            result[i] = self.c
        return result


class MostOftenConstantClassifier(ConstantClassifier):
    def __init__(self):
        pass

    def fit(self, X, y):
        yVals = {}
        for v in y:
            if v not in yVals:
                yVals[v] = 0
            yVals[v] += 1
        mostOften = 0
        mostOftenCount = -1
        for yv, counts in yVals.iteritems():
            if counts > mostOftenCount:
                mostOften = yv
                mostOftenCount = counts
        self.c = mostOften
        return ConstantClassifier.fit(self, X, y)


# results = sentiment_classification(X, y, n_folds=4, classifier=LogisticRegression())
#results = sentiment_classification(X, y, n_folds=4, classifier=ConstantClassifier(1), printing=False)
results = sentiment_classification(X, y, n_folds=4, classifier=MostOftenConstantClassifier(), printing=False)
print 'Accuracy: %s' % np.mean(results['acc'])
print 'F1-measure: %s' % np.mean(results['f1'])
