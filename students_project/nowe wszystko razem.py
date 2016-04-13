from os import path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics, cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import *
import re
import numpy as np
import matplotlib.pyplot as plt

#1. Wyswietlic pod wykresami Accuracy, F1-measure, Predicted. Uśrednic wyniki. - Ola
#2. Sczytać dane ze strony http://www.phonearena.com/phones/. Najlepiej, jakby iterowało po wszystkich telefonach z opiniami. Zapisac do pliku w formacie zdanie-ocena.
#   Oceny mozna troche poroznic, bo wszyscy daja 7 na 10 w górę ;d - Jacek, Ola
#3. Wykorzystac biblioteke spaCy.
#4. Usunac rzadko uzywane slowa.
#5. Jak juz danych bedzie duzo i wszystko bedzie dzialac, to mozna wykorzystac singular value decomposition - kompresja.
#6. Potem ewentualnie pca i inne klasyfikatory, np.naive bayes.

'''import nltk.data
nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')'''

notebook_path = 'C:/Users/Ola1/Desktop/'
test_data_path = "C:/Users/Ola1/Desktop/tekst.txt"
amazon_labels_path = 'C:/Users/Ola1/Desktop/sentiment labelled sentences/amazon_cells_labelled.txt'

y = []
X = []

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
    count_vect = CountVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english')
    data = pd.read_csv(path.join(notebook_path, 'SemEval-2014.csv'), index_col=0)
    docs = data['document'] 
    y = data['sentiment'] # standart name for labels/classes variable 
    X = count_vect.fit_transform(docs)
    return count_vect

def read_amazon_data(amazon_labels_path):
    global X, y
    count_vect = CountVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english')
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
    return count_vect
    
def read_test_data(test_data_path):
    global allData, cameraData, batteryData, screenData, cpuData
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

def transform_test_data(count_vect):
    global allDataX, cameraDataX, batteryDataX, screenDataX, cpuDataX
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

def sentiment_classification_testing(clf, data_type):
    minus = 0
    multiplier = 0
    if data_type == 'amazon':
        minus = 0
        multiplier = 100
    elif data_type == 'twitter':
        minus = 1
        multiplier = 50
    
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

    #for i in range(0, len(predicted)-1):
        #print (allData[i], predicted[i], "\n")
        
    y = np.array([cmMean, scMean, btMean, cpuMean, allMean])
    return y

def plot_prep(arr, titles):
    plt.close('all')
    x = np.array([0,1,2,3,4])
    my_xticks = ["camera","screen","battery","processor", "all"]
    fig, axes = plt.subplots(nrows = 1, ncols = len(arr), sharex = True, figsize=(8*len(arr), 7)) #sharey = True
    plt.setp(axes, xticks = x, xticklabels = my_xticks) #yticks=[0, 20, 40, 60, 80, 100]
    fig.subplots_adjust(wspace=0.4)
    print (type(fig))
    
    i = 0
    for ax in axes:
        ax.set_xlabel('Kategoria')
        ax.set_ylabel('Stopien zadowolenia [%]')
        ax.set_title('Zadowolenie uzytkownikow z poszczegolnych komponentow telefonu.\n'+'Klasyfikator stworzono w oparciu o lableki z: '+titles[i]+'\n')
        ax.bar(x, arr[i], 0.35, color='#F78205')
        i += 1
    plt.show()

'''import tkinter
tk = tkinter.Tk()
canvas = tkinter.Canvas(tk, width=1000, height=500)
canvas.pack()
tk.mainloop()'''

read_test_data(test_data_path)

count_vect = read_amazon_data(amazon_labels_path)
transform_test_data(count_vect)
clf = sentiment_classification_learning(X, y, n_folds=4, classifier=LogisticRegression())
y1 = sentiment_classification_testing(clf, 'amazon')

arr = []
titles = []
arr.append(y1)
titles.append('amazon')

y = []
X = []

count_vect = read_twitter_data(notebook_path)
transform_test_data(count_vect)
clf = sentiment_classification_learning(X, y, n_folds=4, classifier=LogisticRegression())
y2 = sentiment_classification_testing(clf, 'twitter')

arr.append(y2)
titles.append('twitter')
plot_prep(arr, titles)
