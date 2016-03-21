from base1 import *

# tu klep swoj kod....

results = sentiment_classification(X, y, n_folds=4, classifier=LogisticRegression())
print 'Accuracy: %s' % np.mean(results['acc'])
print 'F1-measure: %s' % np.mean(results['f1'])
