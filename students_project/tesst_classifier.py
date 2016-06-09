from students_project.read_data import ReadData
from students_project.core_learning import CoreLearning
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn import linear_model

class TestClassifier:

    pca_ = True

    # learning data
    notebook_path = 'SemEval-2014.csv'
    amazon_labels_path = 'data/amazon_cells_labelled.txt'
    imdb_labels_path = 'data/imdb_labelled.txt'
    yelp_labels_path = 'data/yelp_labelled.txt'
    # scraped_phonearena = "scraped data.txt"
    scraped_phonearena_ALL = "data/short_scraped_labelled.txt"

    # tests data
    test_data_path = "data/random_desc_from_phonearea_unlabelled.txt"
    data_types = ['phonearena', 'amazon', 'imdb', 'yelp', 'twitter']

    # def sentiment_classification_testing(clf, data_type):
    #     minus = 0
    #     multiplier = 1
    #     if data_type == 'amazon':
    #         minus = 0
    #         multiplier = 100
    #     elif data_type == 'twitter':
    #         minus = 1
    #         multiplier = 50
    #     elif data_type == 'phonearena':
    #         minus = 1
    #         multiplier = 100/9
    #     elif data_type == 'all':
    #         minus = 0
    #         multiplier = 50
    #
    #     predicted = clf.predict(screenDataX)
    #     scMean = (np.mean(predicted)-minus)*multiplier
    #     predicted = clf.predict(cameraDataX)
    #     cmMean = (np.mean(predicted)-minus)*multiplier
    #     predicted = clf.predict(batteryDataX)
    #     btMean = (np.mean(predicted)-minus)*multiplier
    #     predicted = clf.predict(cpuDataX)
    #     cpuMean = (np.mean(predicted)-minus)*multiplier
    #     predicted = clf.predict(allDataX)
    #     allMean = (np.mean(predicted)-minus)*multiplier
    #
    #     #for i in range(0, len(predicted)-1):
    #         #print (allData[i], predicted[i], "\n")
    #
    #     y = np.array([cmMean, scMean, btMean, cpuMean, allMean])
    #     return y

    def plot_prep(ys, titles, metricss=None):

        my_xticks = ["camera", "screen", "battery", "processor", "all"]
        width = 0.2  # the width of the bars
        if len(ys) == 3:

            indtmp = np.arange(len(my_xticks))  # the x locations for the groups
            ind = indtmp + 0.0

            fig = plt.figure()
            ax = fig.add_subplot(111)

            colors = ['r', 'y', 'g']
            fig.set_size_inches(15, 8, forward=True)

            if  metricss is not None:
                ax.set_xlabel(metricss)

            ax.set_ylabel('Liczba poszczegolnych ocen')
            ax.set_title('Liczba ocen (0, 1, 2) uzytkownikow dla poszczegolnych komponentow telefonu.\n'+'Klasyfikator stworzono w oparciu o lableki z: '+titles+'\n')
            ax.set_xticks(indtmp + width)
            ax.set_xticklabels(("camera", "screen", "battery", "processor", "all"))

            rects0 = ax.bar(ind, ys[0], width, color=colors[0])
            ind += width
            rects1 = ax.bar(ind, ys[1], width, color=colors[1])
            ind += width
            rects2 = ax.bar(ind, ys[2], width, color=colors[2])
            ax.legend((rects0[0], rects1[0], rects2[0]), ('negatywne', 'neutralne', 'pozytywne'))
        elif len(ys) == 2:

            indtmp = np.arange(len(my_xticks))  # the x locations for the groups
            ind = indtmp + 0.0

            fig = plt.figure()
            ax = fig.add_subplot(111)

            colors = ['r', 'g']
            fig.set_size_inches(15, 8, forward=True)

            if metricss is not None:
                ax.set_xlabel(metricss)

            ax.set_ylabel('Liczba poszczegolnych ocen')
            ax.set_title('Liczba ocen pozytywnych i negatywnych uzytkownikow dla poszczegolnych komponentow telefonu.\n' + 'Klasyfikator stworzono w oparciu o lableki z: ' + titles + '\n')
            ax.set_xticks(indtmp + width)
            ax.set_xticklabels(("camera", "screen", "battery", "processor", "all"))

            rects0 = ax.bar(ind, ys[0], width, color=colors[0])
            ind += width
            rects1 = ax.bar(ind, ys[1], width, color=colors[1])
            ax.legend((rects0[0], rects1[0]), ('negatywne', 'pozytywne'))

        else:
            x = np.array([0, 1, 2, 3, 4])

            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.setp(ax, xticks=x, xticklabels=my_xticks)
            fig.set_size_inches(15, 8, forward=True)

            if metricss is not None:
                ax.set_xlabel(metricss)
            ax.set_ylabel('Srednia ocena uzytkownika')
            ax.set_title('\n'+'Srednie oceny uzytkownikow dla poszczegolnych komponentow telefonu.\n'+'Klasyfikator stworzono w oparciu o lableki z: '+titles+'\n')
            ax.bar(x, ys, width+0.15, color='#F78205')
        plt.show()

    # def learnAndTestOnEveryDatasetSeparately():
    #     read_test_data(test_data_path)
    #     ys = []
    #     titles = []
    #
    #     global allDataX, cameraDataX, batteryDataX, screenDataX, cpuDataX
    #     metricss = []
    #     numOfIters = 15
    #
    #     ys_a = []
    #     ys_t = []
    #     ys_ph = []
    #
    #     metricss_a = []
    #     metricss_t = []
    #     metricss_ph = []
    #
    #     titles.append('amazon')
    #     titles.append('twitter')
    #     titles.append('phonearena')
    #
    #     count_vect_a, X_a, Y_a = read_structured_data(amazon_labels_path)
    #     count_vect_t, X_t, Y_t = read_twitter_data(notebook_path)
    #     count_vect_ph, X_ph, Y_ph = read_structured_data(scraped_phonearena)
    #
    #     allDataX_a, cameraDataX_a, batteryDataX_a, screenDataX_a, cpuDataX_a = transform_test_data(count_vect_a)
    #     allDataX_t, cameraDataX_t, batteryDataX_t, screenDataX_t, cpuDataX_t = transform_test_data(count_vect_t)
    #     allDataX_ph, cameraDataX_ph, batteryDataX_ph, screenDataX_ph, cpuDataX_ph = transform_test_data(count_vect_ph)
    #
    #     if pca_:
    #         pcaModel, X_a = pca_opt(X_a)
    #         pca_tmp, allDataX_a = pca_opt(allDataX_a, pcaModel)
    #         pca_tmp, cameraDataX_a = pca_opt(cameraDataX_a, pcaModel)
    #         pca_tmp, batteryDataX_a = pca_opt(batteryDataX_a, pcaModel)
    #         pca_tmp, screenDataX_a = pca_opt(screenDataX_a, pcaModel)
    #         pca_tmp, cpuDataX_a = pca_opt(cpuDataX_a, pcaModel)
    #
    #     for i in range(0, numOfIters):
    #         clf, metricss_ = sentiment_classification_learning(X_a, Y_a, n_folds=4, classifier=LogisticRegression())
    #         allDataX, cameraDataX, batteryDataX, screenDataX, cpuDataX = allDataX_a, cameraDataX_a, batteryDataX_a, screenDataX_a, cpuDataX_a
    #         y1 = sentiment_classification_testing(clf, 'amazon')
    #         ys_a.append(y1)
    #         metricss_a.append(metricss_)
    #
    #         clf, metricss_ = sentiment_classification_learning(X_t, Y_t, n_folds=4, classifier=LogisticRegression())
    #         allDataX, cameraDataX, batteryDataX, screenDataX, cpuDataX = allDataX_t, cameraDataX_t, batteryDataX_t, screenDataX_t, cpuDataX_t
    #         y2 = sentiment_classification_testing(clf, 'twitter')
    #         ys_t.append(y2)
    #         metricss_t.append(metricss_)
    #
    #         clf, metricss_ = sentiment_classification_learning(X_ph, Y_ph, n_folds=6, classifier=LogisticRegression())
    #         allDataX, cameraDataX, batteryDataX, screenDataX, cpuDataX = allDataX_ph, cameraDataX_ph, batteryDataX_ph, screenDataX_ph, cpuDataX_ph
    #         y3 = sentiment_classification_testing(clf, 'phonearena')
    #         ys_ph.append(y3)
    #         metricss_ph.append(metricss_)
    #
    #     ys.append([sum(x)/len(ys_a) for x in zip(*ys_a)])
    #     ys.append([sum(x)/len(ys_t) for x in zip(*ys_t)])
    #     ys.append([sum(x)/len(ys_ph) for x in zip(*ys_ph)])
    #
    #     metricss.append([sum(x)/len(metricss_a) for x in zip(*metricss_a)])
    #     metricss.append([sum(x)/len(metricss_t) for x in zip(*metricss_t)])
    #     metricss.append([sum(x)/len(metricss_ph) for x in zip(*metricss_ph)])
    #
    #     plot_prep(ys, titles, metricss)

    # przyjmowane wartosci ocen dla zdan: [0, 1, 2]. Uczy na polaczonych danych dzieki normalizacji danych metoda read_data_with_normalization.
    # def learnAndTestOnJoinedData():
    #     read_test_data(test_data_path)
    #     ys = []
    #     titles = []
    #
    #     global allDataX, cameraDataX, batteryDataX, screenDataX, cpuDataX
    #     metricss = []
    #
    #     files = [[amazon_labels_path, data_types[1]], [imdb_labels_path, data_types[2]], [yelp_labels_path, data_types[3]], [notebook_path, data_types[4]]]
    #     count_vect, X, Y = read_data_with_normalization(files)
    #     allDataX_a, cameraDataX_a, batteryDataX_a, screenDataX_a, cpuDataX_a = transform_test_data(count_vect)
    #     clf, metricss_ = sentiment_classification_learning(X, Y, n_folds=10, classifier=LogisticRegression())
    #     allDataX, cameraDataX, batteryDataX, screenDataX, cpuDataX = allDataX_a, cameraDataX_a, batteryDataX_a, screenDataX_a, cpuDataX_a
    #     y = sentiment_classification_testing(clf, 'all')
    #     ys.append(y)
    #     metricss.append(metricss_)
    #     titles.append('all')
    #     plot_prep(ys, titles, metricss)

    # przyjmowane wartosci ocen dla zdan: [0, 1, 2]
    # pierwszy etap: [1, [0, 2]=0]
    # drugi etap: if not neutral (1) then test: 0 or 2 (zamiast 2 wstawiam 1, bo sie klasyfikator wywala, a potem znowu konwertuje na 2...)
    # odrzucam na razie tweety, bo sa zbyt dlugie i trzeba by je jakos podzielic (?)


    def twoStageClassification(pocoargument=None):

        allData, cameraData, batteryData, screenData, cpuData = ReadData.read_test_data(TestClassifier.test_data_path)

        files = [[TestClassifier.amazon_labels_path, TestClassifier.data_types[1]], [TestClassifier.imdb_labels_path, TestClassifier.data_types[2]], [TestClassifier.yelp_labels_path, TestClassifier.data_types[3]], [TestClassifier.scraped_phonearena_ALL, TestClassifier.data_types[0]]]
        stats, docs, Y = ReadData.read_data_with_normalization(files, return_stats=True)

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

        # II. etap
        pos_to_neg_equalization = int(stats[2] / stats[0] * 100)
        # I. etap
        pos_to_neut_equalization = int(stats[1] * 2 / stats[0] * 100)
        neg_to_neut_equalization = int(stats[1] * 2 / stats[2] * 100)

        if len(docs) != len(Y):
            print ("BLAD!!! Cos przy wczytywaniu danych poszlo nie tak. Metoda: read_data_with_normalization().")
        else:
            for i in range(0, len(docs)):
                if Y[i] == 0 or Y[i] == 2:
                    # zamiast 2 wstawiam 1, bo sie klasyfikator wywala, a potem znowu konwertuje na 2...
                    if Y[i] == 2:
                        if random.randint(0, 100) < pos_to_neg_equalization:
                            Y2.append(Y[i]-1)
                            Xdocs2.append(docs[i])
                        if random.randint(0, 100) < pos_to_neut_equalization:
                            # [0, 2]=0
                            Y1.append(0)
                            Xdocs1.append(docs[i])
                    else:
                        Y2.append(Y[i])
                        Xdocs2.append(docs[i])
                        if random.randint(0, 100) < neg_to_neut_equalization:
                            # [0, 2]=0
                            Y1.append(0)
                            Xdocs1.append(docs[i])
                elif Y[i] == 1:
                    Y1.append(1)
                    Xdocs1.append(docs[i])
                else:
                    print("Blad!!! Niepoprawna ocena (", Y[i], ") zdania: ", docs[i])

        # if len(docs) != len(Y):
        #     print("BLAD!!! Cos przy wczytywaniu danych poszlo nie tak. Metoda: read_data_with_normalization().")
        # else:
        #     for i in range(0, len(docs)):
        #         if Y[i] == 0 or Y[i] == 2:
        #             # zamiast 2 wstawiam 1, bo sie klasyfikator wywala, a potem znowu konwertuje na 2...
        #             if Y[i] == 2:
        #                 Y2.append(Y[i] - 1)
        #             else:
        #                 Y2.append(Y[i])
        #             Xdocs2.append(docs[i])
        #             # [0, 2]=0
        #             Y1.append(0)
        #             Xdocs1.append(docs[i])
        #         elif Y[i] == 1:
        #             Y1.append(1)
        #             Xdocs1.append(docs[i])
        #         else:
        #             print("Blad!!! Niepoprawna ocena (", Y[i], ") zdania: ", docs[i])
        # # wyrownanie liczby pozytywnych i negatywnych danych uczacych dla etapu drugiego tylko i wylacznie -> Xdocs2, Y2
        # diff = stats[0]-stats[2]
        # for iter in range(0, diff):
        #     change = False
        #     while not change:
        #         r = random.randint(0, len(Y2)-1)
        #         if Y2[r] == 1:
        #             del Y2[r]
        #             del Xdocs2[r]
        #             change = True


            # etap I.

            X1, count_vect =  ReadData.get_ngrams_and_countVect(Xdocs1)
            allDataX1, cameraDataX1, batteryDataX1, screenDataX1, cpuDataX1 = ReadData.transform_test_data(count_vect, allData, cameraData, batteryData, screenData, cpuData)

            print ("Etap pierwszy. Dlugosc danych uczacych: ", X1.shape[0])
            clf, metricss_ = CoreLearning.sentiment_classification_learning(X1, np.array(Y1), n_folds=15, classifier=linear_model.LogisticRegression())
            print ('Accuracy: %s, F1-measure: %s, Predicted: %s' % (round(metricss_[0], 3), round(metricss_[1], 3), round(metricss_[2], 3)))
            metricss = 'Etap I: Accuracy: %s, F1-measure: %s, Predicted: %s' % (round(metricss_[0], 3), round(metricss_[1], 3), round(metricss_[2], 3))

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
            X2, count_vect = ReadData.get_ngrams_and_countVect(Xdocs2)

            allDataX2, cameraDataX2, batteryDataX2, screenDataX2, cpuDataX2 = ReadData.transform_test_data(count_vect, allData, cameraData, batteryData, screenData, cpuData)

            print("Etap drugi. Dlugosc danych uczacych: ", X2.shape[0])
            clf, metricss_ = CoreLearning.sentiment_classification_learning(X2, np.array(Y2), n_folds=15, classifier=linear_model.LogisticRegression())
            print('Accuracy: %s, F1-measure: %s, Predicted: %s' % (round(metricss_[0], 3), round(metricss_[1], 3), round(metricss_[2], 3)))
            metricss += '\nEtap II: Accuracy: %s, F1-measure: %s, Predicted: %s' % (round(metricss_[0], 3), round(metricss_[1], 3), round(metricss_[2], 3))

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

            TestClassifier.plot_prep([np.mean(cameraDataYresults), np.mean(screenDataYresults), np.mean(batteryDataYresults), np.mean(cpuDataYresults), np.mean(allDataYresults)], "Phonearena, imdb, amazon, yelp")

            negposnut = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
            results = [cameraDataYresults, screenDataYresults, batteryDataYresults, cpuDataYresults, allDataYresults]

            for i in range(0, len(results)):
                negposnut[0][i] = results[i].count(0)
                negposnut[1][i] = results[i].count(1)
                negposnut[2][i] = results[i].count(2)

            TestClassifier.plot_prep(negposnut, "Phonearena, imdb, amazon, yelp", metricss)

    def oneStageClassification(pocoargument=None, pcaopt=False):

        allData, cameraData, batteryData, screenData, cpuData = ReadData.read_test_data(TestClassifier.test_data_path)

        files = [[TestClassifier.amazon_labels_path, TestClassifier.data_types[1]],[TestClassifier.imdb_labels_path, TestClassifier.data_types[2]],[TestClassifier.yelp_labels_path, TestClassifier.data_types[3]]]
        stats, docs, Y = ReadData.read_data_with_normalization(files, return_stats=True)
        Y = [y/2 for y in Y.tolist()]

        X, count_vect = ReadData.get_ngrams_and_countVect(docs)
        allData, cameraData, batteryData, screenData, cpuData = ReadData.transform_test_data(count_vect, allData, cameraData, batteryData, screenData, cpuData)

        if pcaopt:
            pca_model, X = ReadData.pca_opt(X)
            print ('1')
            pca_model_tmp, allData = ReadData.pca_opt(allData, pca_model)
            print('2')
            pca_model_tmp, cameraData = ReadData.pca_opt(cameraData, pca_model)
            print('3')
            pca_model_tmp, batteryData = ReadData.pca_opt(batteryData, pca_model)
            print('4')
            pca_model_tmp, screenData  = ReadData.pca_opt(screenData, pca_model)
            print('5')
            pca_model_tmp, cpuData = ReadData.pca_opt(cpuData, pca_model)


        print("Dlugosc danych uczacych: ", X.shape[0])
        clf, metricss_ = CoreLearning.sentiment_classification_learning(X, np.array(Y), n_folds=15, classifier=linear_model.LogisticRegression())
        print('Accuracy: %s, F1-measure: %s, Predicted: %s' % (round(metricss_[0], 3), round(metricss_[1], 3), round(metricss_[2], 3)))
        metricss = 'Accuracy: %s, F1-measure: %s, Predicted: %s' % (round(metricss_[0], 3), round(metricss_[1], 3), round(metricss_[2], 3))

        allDataYresults = clf.predict(allData).tolist()
        cameraDataYresults = clf.predict(cameraData).tolist()
        batteryDataYresults = clf.predict(batteryData).tolist()
        screenDataYresults = clf.predict(screenData).tolist()
        cpuDataYresults = clf.predict(cpuData).tolist()

        TestClassifier.plot_prep([np.mean(cameraDataYresults), np.mean(screenDataYresults), np.mean(batteryDataYresults),np.mean(cpuDataYresults), np.mean(allDataYresults)], "Imdb, amazon, yelp")

        negposnut = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        results = [cameraDataYresults, screenDataYresults, batteryDataYresults, cpuDataYresults, allDataYresults]

        for i in range(0, len(results)):
            negposnut[0][i] = results[i].count(0)
            negposnut[1][i] = results[i].count(1)

        TestClassifier.plot_prep(negposnut, "Imdb, amazon, yelp", metricss)


if __name__ == '__main__':
    TestClassifier.twoStageClassification()
    TestClassifier.oneStageClassification()
    TestClassifier.oneStageClassification(pcaopt=True)
