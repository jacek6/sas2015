from base1 import *

class MyCls:
    def fit(self, X, y):
        self.yy = y
        return self
    
    def predict(self, X):
        result = np.zeros(shape=(X.shape[0]))
        for i in range(0, result.size-1):
            result[i] = 2
        return result
        
#results = sentiment_classification(X, y, n_folds=4, classifier=LogisticRegression())
results = sentiment_classification(X, y, n_folds=4, classifier=MyCls(), printing=False)
print 'Accuracy: %s' % np.mean(results['acc'])
print 'F1-measure: %s' % np.mean(results['f1'])
