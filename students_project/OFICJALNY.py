import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import math
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn import metrics, cross_validation
    from sklearn.linear_model import *
    from sklearn.decomposition import PCA
    from sklearn.decomposition import IncrementalPCA

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

# learning data
notebook_path = 'SemEval-2014.csv'
amazon_labels_path = 'amazon_cells_labelled.txt'
imdb_labels_path = 'imdb_labelled.txt'
yelp_labels_path = 'yelp_labelled.txt'
scraped_phonearena = "scraped data.txt"
scraped_phonearena_ALL = "scraped all data.txt"
# tests data
test_data_path = "tekst.txt"
data_types = ['phonearena', 'amazon', 'imdb', 'yelp', 'twitter']

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

# optymalizacje
countVec_min_df = 1
pca_ = True
svdOpt = 10

def read_twitter_data(notebook_path):
    count_vect = CountVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english', min_df = countVec_min_df)
    data = pd.read_csv(notebook_path, index_col=0)
    docs = data['document']
    y = data['sentiment'] # standart name for labels/classes variable
    X = count_vect.fit_transform(docs)
    return count_vect, X, y

def read_structured_data(labels_path):
    count_vect = CountVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english', min_df = countVec_min_df)
    docs = []
    y = []
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if len(line) > 7:
                sentim = line[len(line)-1]
                try:
                    sen = int(sentim)
                    line = line[:-1].strip()
                    try:
                        sen_ = int(line[len(line)-1])
                        sen += sen_ * 10 # 1*10 + 0 = 10
                        line = line[:-1].strip()
                    except:
                        pass
                    if sen < 11 and sen > -1:
                        y.append(sen)
                        docs.append(line)
                except:
                    pass
    Y = np.array(y)
    X = count_vect.fit_transform(docs)
    return count_vect, X, Y

def read_data_with_normalization(pathWithTypeOfData, return_count_vect = True):
    count_vect = CountVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english', min_df = countVec_min_df)
    docs = []
    y = []
    for path in pathWithTypeOfData:
        if path[1] == 'twitter':
            data = pd.read_csv(path[0], index_col=0)
            docs_tmp = data['document']
            y_tmp = data['sentiment']
            if len(docs_tmp) == len(y_tmp):
                docs += list(docs_tmp)
                y += [yy-1 for yy in list(y_tmp)] # oceny [1, 2, 3] -> [0, 1, 2]
        elif path[1] == 'amazon' or path[1] == 'imdb' or path[1] == 'yelp' or path[1] == 'phonearena':
            with open(path[0]) as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 8:
                        sentim = line[len(line)-1]
                        try:
                            sen = int(sentim)
                            line = line[:-1].strip()
                            try:
                                sen_ = int(line[len(line)-1])
                                sen += sen_ * 10
                                line = line[:-1].strip()
                            except:
                                pass
                            if sen < 11 and sen > -1:
                                if path[1] == 'phonearena':
                                    # sen = math.log(sen, 2.154) - 1
                                    if sen >= 8:
                                        sen = 2
                                    elif sen >=4:
                                        sen = 1
                                    else:
                                        sen = 0
                                else:
                                    # nie ma wartosci neutrealnych dla labelek z amazona, imdb, yelp
                                    if sen == 1:
                                        sen += 1
                                y.append(sen)
                                docs.append(line)
                        except:
                            pass
        Y = np.array(y)
        X = count_vect.fit_transform(docs)
    if return_count_vect:
        return count_vect, X, Y
    else:
        return docs, Y

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
    newVec = CountVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english', vocabulary=count_vect.vocabulary_)
    allDataX = newVec.fit_transform(allData)
    cameraDataX = newVec.fit_transform(cameraData)
    batteryDataX = newVec.fit_transform(batteryData)
    screenDataX = newVec.fit_transform(screenData)
    cpuDataX = newVec.fit_transform(cpuData)

    return allDataX, cameraDataX, batteryDataX, screenDataX, cpuDataX

# returns pcaModel [type: PCA] (if not existed before pca_opt) and optimised array [type: sparse matrix]
def pca_opt(sparseArray, pcaModel=None):
    array = sparseArray.toarray()
    if pcaModel is None:
        newSize = len(array[0])/svdOpt
        if len(array) < 2900:
            pca = PCA(n_components=newSize)
            Xtransformed = pca.fit_transform(array)
            Xtransformed = sparse.csr_matrix(Xtransformed)
        else:
            chunkSize = 2000
            chunks = [array[i:i+chunkSize] for i in range(0, len(array), chunkSize)]
            pca = IncrementalPCA(n_components=newSize)

            for chunk in chunks:
                pca.partial_fit(chunk)

            Xtransformed = None
            for chunk in chunks:
                Xchunk = pca.transform(chunk)
                if Xtransformed == None:
                    Xtransformed = Xchunk
                else:
                    Xtransformed = np.vstack((Xtransformed, Xchunk))
            Xtransformed = sparse.csr_matrix(Xtransformed)

        return pca, Xtransformed
    else:
        if len(array) < 2900:
            Xtransformed = pcaModel.transform(array)
            Xtransformed = sparse.csr_matrix(Xtransformed)
        else:
            chunkSize = 2000
            chunks = [array[i:i+chunkSize] for i in range(0, len(array), chunkSize)]

            Xtransformed = None
            for chunk in chunks:
                Xchunk = pcaModel.transform(chunk)
                if Xtransformed == None:
                    Xtransformed = Xchunk
                else:
                    Xtransformed = np.vstack((Xtransformed, Xchunk))
            Xtransformed = sparse.csr_matrix(Xtransformed)

        return pcaModel, Xtransformed

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

        #print ('Accuracy: %s, F1-measure: %s, Predicted: %s' % (np.mean(results['acc']), np.mean(results['f1']), np.mean(results['prec'])))
        metricss = [np.mean(results['acc']), np.mean(results['f1']), np.mean(results['prec'])]
        return clf, metricss

def sentiment_classification_testing(clf, data_type):
    minus = 0
    multiplier = 1
    if data_type == 'amazon':
        minus = 0
        multiplier = 100
    elif data_type == 'twitter':
        minus = 1
        multiplier = 50
    elif data_type == 'phonearena':
        minus = 1
        multiplier = 100/9
    elif data_type == 'all':
        minus = 0
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

def plot_prep(ys, titles, metricss=None):
    plt.close('all')
    x = np.array([0,1,2,3,4])
    my_xticks = ["camera","screen","battery","processor", "all"]
    fig, axes = plt.subplots(nrows = 1, ncols = len(ys), sharex = True, figsize=(8*len(ys), 7)) #sharey = True
    plt.setp(axes, xticks = x, xticklabels = my_xticks) #yticks=[0, 20, 40, 60, 80, 100]
    fig.subplots_adjust(wspace=0.4)
    print (type(fig))

    i = 0
    # jakby cos sie sypalo, to wywalic tego ifa
    if len(ys) > 1:
        for ax in axes:
            if  metricss is not None:
                ax.set_xlabel('Kategoria'+'\n'+'Accuracy: %s, F1-measure: %s, Predicted: %s' % (round(metricss[i][0], 3), round(metricss[i][1], 3), round(metricss[i][2], 3)))
                ax.set_ylabel('Stopien zadowolenia [%]')
            ax.set_title('Zadowolenie uzytkownikow z poszczegolnych komponentow telefonu.\n'+'Klasyfikator stworzono w oparciu o lableki z: '+titles[i]+'\n')
            ax.bar(x, ys[i], 0.35, color='#F78205')
            i += 1
    else:
        if metricss is not None:
            axes.set_xlabel('Kategoria' + '\n' + 'Accuracy: %s, F1-measure: %s, Predicted: %s' % (round(metricss[i][0], 3), round(metricss[i][1], 3), round(metricss[i][2], 3)))
            axes.set_ylabel('Stopien zadowolenia [%]')
        axes.set_title('Zadowolenie uzytkownikow z poszczegolnych komponentow telefonu.\n' + 'Klasyfikator stworzono w oparciu o lableki z: ' +titles[0] + '\n')
        axes.bar(x, ys[0], 0.35, color='#F78205')
    plt.show()

'''import tkinter
tk = tkinter.Tk()
canvas = tkinter.Canvas(tk, width=1000, height=500)
canvas.pack()
tk.mainloop()'''

def learnAndTestOnEveryDatasetSeparately():
    read_test_data(test_data_path)
    ys = []
    titles = []

    global allDataX, cameraDataX, batteryDataX, screenDataX, cpuDataX
    metricss = []
    numOfIters = 15

    ys_a = []
    ys_t = []
    ys_ph = []

    metricss_a = []
    metricss_t = []
    metricss_ph = []

    titles.append('amazon')
    titles.append('twitter')
    titles.append('phonearena')

    count_vect_a, X_a, Y_a = read_structured_data(amazon_labels_path)
    count_vect_t, X_t, Y_t = read_twitter_data(notebook_path)
    count_vect_ph, X_ph, Y_ph = read_structured_data(scraped_phonearena)

    allDataX_a, cameraDataX_a, batteryDataX_a, screenDataX_a, cpuDataX_a = transform_test_data(count_vect_a)
    allDataX_t, cameraDataX_t, batteryDataX_t, screenDataX_t, cpuDataX_t = transform_test_data(count_vect_t)
    allDataX_ph, cameraDataX_ph, batteryDataX_ph, screenDataX_ph, cpuDataX_ph = transform_test_data(count_vect_ph)

    if pca_:
        pcaModel, X_a = pca_opt(X_a)
        pca_tmp, allDataX_a = pca_opt(allDataX_a, pcaModel)
        pca_tmp, cameraDataX_a = pca_opt(cameraDataX_a, pcaModel)
        pca_tmp, batteryDataX_a = pca_opt(batteryDataX_a, pcaModel)
        pca_tmp, screenDataX_a = pca_opt(screenDataX_a, pcaModel)
        pca_tmp, cpuDataX_a = pca_opt(cpuDataX_a, pcaModel)

    for i in range(0, numOfIters):
        clf, metricss_ = sentiment_classification_learning(X_a, Y_a, n_folds=4, classifier=LogisticRegression())
        allDataX, cameraDataX, batteryDataX, screenDataX, cpuDataX = allDataX_a, cameraDataX_a, batteryDataX_a, screenDataX_a, cpuDataX_a
        y1 = sentiment_classification_testing(clf, 'amazon')
        ys_a.append(y1)
        metricss_a.append(metricss_)

        clf, metricss_ = sentiment_classification_learning(X_t, Y_t, n_folds=4, classifier=LogisticRegression())
        allDataX, cameraDataX, batteryDataX, screenDataX, cpuDataX = allDataX_t, cameraDataX_t, batteryDataX_t, screenDataX_t, cpuDataX_t
        y2 = sentiment_classification_testing(clf, 'twitter')
        ys_t.append(y2)
        metricss_t.append(metricss_)

        clf, metricss_ = sentiment_classification_learning(X_ph, Y_ph, n_folds=6, classifier=LogisticRegression())
        allDataX, cameraDataX, batteryDataX, screenDataX, cpuDataX = allDataX_ph, cameraDataX_ph, batteryDataX_ph, screenDataX_ph, cpuDataX_ph
        y3 = sentiment_classification_testing(clf, 'phonearena')
        ys_ph.append(y3)
        metricss_ph.append(metricss_)

    ys.append([sum(x)/len(ys_a) for x in zip(*ys_a)])
    ys.append([sum(x)/len(ys_t) for x in zip(*ys_t)])
    ys.append([sum(x)/len(ys_ph) for x in zip(*ys_ph)])

    metricss.append([sum(x)/len(metricss_a) for x in zip(*metricss_a)])
    metricss.append([sum(x)/len(metricss_t) for x in zip(*metricss_t)])
    metricss.append([sum(x)/len(metricss_ph) for x in zip(*metricss_ph)])

    plot_prep(ys, titles, metricss)

# przyjmowane wartosci ocen dla zdan: [0, 1, 2]. Uczy na polaczonych danych dzieki normalizacji danych metoda read_data_with_normalization.
def learnAndTestOnJoinedData():
    read_test_data(test_data_path)
    ys = []
    titles = []

    global allDataX, cameraDataX, batteryDataX, screenDataX, cpuDataX
    metricss = []

    files = [[amazon_labels_path, data_types[1]], [imdb_labels_path, data_types[2]], [yelp_labels_path, data_types[3]], [notebook_path, data_types[4]]]
    count_vect, X, Y = read_data_with_normalization(files)
    allDataX_a, cameraDataX_a, batteryDataX_a, screenDataX_a, cpuDataX_a = transform_test_data(count_vect)
    clf, metricss_ = sentiment_classification_learning(X, Y, n_folds=10, classifier=LogisticRegression())
    allDataX, cameraDataX, batteryDataX, screenDataX, cpuDataX = allDataX_a, cameraDataX_a, batteryDataX_a, screenDataX_a, cpuDataX_a
    y = sentiment_classification_testing(clf, 'all')
    ys.append(y)
    metricss.append(metricss_)
    titles.append('all')
    plot_prep(ys, titles, metricss)

# przyjmowane wartosci ocen dla zdan: [0, 1, 2]
# pierwszy etap: [1, [0, 2]=0]
# drugi etap: if not neutral (1) then test: 0 or 2 (zamiast 2 wstawiam 1, bo sie klasyfikator wywala, a potem znowu konwertuje na 2...)
# odrzucam na razie tweety, bo sa zbyt dlugie i trzeba by je jakos podzielic (?)
def twoStageClassification():

    global allData, cameraData, batteryData, screenData, cpuData

    read_test_data(test_data_path)

    files = [[amazon_labels_path, data_types[1]], [imdb_labels_path, data_types[2]], [yelp_labels_path, data_types[3]], [scraped_phonearena_ALL, data_types[0]]]
    docs, Y = read_data_with_normalization(files, return_count_vect=False)

    # split data for two stages
    Xdocs1 = []
    Xdocs2 = []
    Y1 = []
    Y2 = []

    allDataYresults = []
    cameraDataYresults = []
    batteryDataYresults = []
    screenDataYresults = []
    cpuDataYresults = []


    if len(docs) != len(Y):
        print ("BLAD!!! Cos przy wczytywaniu danych poszlo nie tak. Metoda: read_data_with_normalization().")
    else:
        for i in range(0, len(docs)):
            if Y[i] == 0 or Y[i] == 2:
                # zamiast 2 wstawiam 1, bo sie klasyfikator wywala, a potem znowu konwertuje na 2...
                if Y[i] == 2:
                    Y2.append(Y[i]-1)
                else:
                    Y2.append(Y[i])
                Xdocs2.append(docs[i])
                # [0, 2]=0
                Y1.append(0)
                Xdocs1.append(docs[i])
            elif Y[i] == 1:
                Y1.append(1)
                Xdocs1.append(docs[i])
            else:
                print("Blad!!! Niepoprawna ocena (", Y[i], ") zdania: ", docs[i])

        # etap I.
        count_vect = CountVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english', min_df=countVec_min_df)
        X1 = count_vect.fit_transform(Xdocs1)

        allDataX1, cameraDataX1, batteryDataX1, screenDataX1, cpuDataX1 = transform_test_data(count_vect)

        print ("Etap pierwszy. Dlugosc danych uczacych: ", X1.getnnz())
        clf, metricss_ = sentiment_classification_learning(X1, np.array(Y1), n_folds=15, classifier=LogisticRegression())
        print ('Accuracy: %s, F1-measure: %s, Predicted: %s' % (round(metricss_[0], 3), round(metricss_[1], 3), round(metricss_[2], 3)))

        # [1, [0, 2]=0]
        allDataY1 = clf.predict(allDataX1)
        cameraDataY1 = clf.predict(cameraDataX1)
        batteryDataY1 = clf.predict(batteryDataX1)
        screenDataY1 = clf.predict(screenDataX1)
        cpuDataY1 = clf.predict(cpuDataX1)


        # etap posredniczacy I. i II.
        # if neutral (1) then append and delete from the global test data
        for i in range(0, len(allDataY1)):
            if allDataY1[i] == 1:
                allDataYresults.append(1)
                np.delete(allData, i)

        for i in range(0, len(cameraDataY1)):
            if cameraDataY1[i] == 1:
                cameraDataYresults.append(1)
                np.delete(cameraData, i)

        for i in range(0, len(batteryDataY1)):
            if batteryDataY1[i] == 1:
                batteryDataYresults.append(1)
                np.delete(batteryData, i)

        for i in range(0, len(screenDataY1)):
            if screenDataY1[i] == 1:
                screenDataYresults.append(1)
                np.delete(screenData, i)

        for i in range(0, len(cpuDataY1)):
            if cpuDataY1[i] == 1:
                cpuDataYresults.append(1)
                np.delete(cpuData, i)

        # etap II.
        count_vect = CountVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english', min_df=countVec_min_df)
        X2 = count_vect.fit_transform(Xdocs2)

        # globalne zmienne allData itp. zostaly juz zaktualizowane
        allDataX2, cameraDataX2, batteryDataX2, screenDataX2, cpuDataX2 = transform_test_data(count_vect)

        print("Etap drugi. Dlugosc danych uczacych: ", X2.getnnz())
        clf, metricss_ = sentiment_classification_learning(X2, np.array(Y2), n_folds=15, classifier=LogisticRegression())
        print('Accuracy: %s, F1-measure: %s, Predicted: %s' % (round(metricss_[0], 3), round(metricss_[1], 3), round(metricss_[2], 3)))

        # if not neutral (1) then test: 0 or 2 (zamiast 2 mam 1, bo inaczej sie klasyfikator wywala, a potem znowu konwertuje na 2...)
        allDataY2 = clf.predict(allDataX2)
        cameraDataY2 = clf.predict(cameraDataX2)
        batteryDataY2 = clf.predict(batteryDataX2)
        screenDataY2 = clf.predict(screenDataX2)
        cpuDataY2 = clf.predict(cpuDataX2)


        allDataYresults += [x*2 for x in allDataY2.tolist()]
        cameraDataYresults += [x*2 for x in cameraDataY2.tolist()]
        batteryDataYresults += [x*2 for x in batteryDataY2.tolist()]
        screenDataYresults += [x*2 for x in screenDataY2.tolist()]
        cpuDataYresults += [x*2 for x in cpuDataY2.tolist()]

        plot_prep([[np.mean(allDataYresults)*50, np.mean(cameraDataYresults)*50, np.mean(batteryDataYresults)*50, np.mean(screenDataYresults)*50, np.mean(cpuDataYresults)*50]], ["Phonearena, imdb, amazon, yelp"])


if __name__ == '__main__':
    twoStageClassification()
