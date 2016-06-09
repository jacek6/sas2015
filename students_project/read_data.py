import pandas as pd
import numpy as np
import re
from scipy import sparse
import warnings
with warnings.catch_warnings():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import PCA
    from sklearn.decomposition import IncrementalPCA

class ReadData:

    # optymalizacje
    countVec_min_df = 1
    svdOpt = 10
    ngram_max = 2

    def get_ngrams_and_countVect(text_data):
        count_vect = CountVectorizer(ngram_range=(1, ReadData.ngram_max), lowercase=True, stop_words='english', min_df=ReadData.countVec_min_df)
        X = count_vect.fit_transform(text_data)
        return X,count_vect

    def get_ngrams(text_data, cntvect):
        newVec = CountVectorizer(ngram_range=(1, ReadData.ngram_max), lowercase=True, stop_words='english', vocabulary=cntvect.vocabulary_)
        X = newVec.fit_transform(text_data)
        return X

    def read_twitter_data(notebook_path):
        """
        :param notebook_path:
        :return: CountVectorizer,
        """
        data = pd.read_csv(notebook_path, index_col=0)
        docs = data['document']
        y = data['sentiment']  # standart name for labels/classes variable
        return docs, y

    def read_structured_data(labels_path):
        docs = []
        y = []
        with open(labels_path) as f:
            for line in f:
                line = line.strip()
                if len(line) > 7:
                    sentim = line[len(line) - 1]
                    try:
                        sen = int(sentim)
                        line = line[:-1].strip()
                        try:
                            sen_ = int(line[len(line) - 1])
                            sen += sen_ * 10  # 1*10 + 0 = 10
                            line = line[:-1].strip()
                        except:
                            pass
                        if sen < 11 and sen > -1:
                            y.append(sen)
                            docs.append(line)
                    except:
                        pass
        Y = np.array(y)
        return docs, Y

    def read_data_with_normalization(pathWithTypeOfData, return_stats=False):
        docs = []
        y = []
        positive_counter = 0
        negative_counter = 0
        neutral_counter = 0
        positive_counter_all = 0
        negative_counter_all = 0
        neutral_counter_all = 0
        for path in pathWithTypeOfData:
            if path[1] == 'twitter':
                data = pd.read_csv(path[0], index_col=0)
                docs_tmp = data['document']
                y_tmp = data['sentiment']
                if len(docs_tmp) == len(y_tmp):
                    docs += list(docs_tmp)
                    y += [yy - 1 for yy in list(y_tmp)]  # oceny [1, 2, 3] -> [0, 1, 2]
            elif path[1] == 'amazon' or path[1] == 'imdb' or path[1] == 'yelp' or path[1] == 'phonearena':
                with open(path[0], encoding="utf8") as f:
                    for line in f:
                        line = line.strip()
                        if len(line) > 8:
                            sentim = line[len(line) - 1]
                            try:
                                sen = int(sentim)
                                line = line[:-1].strip()
                                try:
                                    sen_ = int(line[len(line) - 1])
                                    sen += sen_ * 10
                                    line = line[:-1].strip()
                                except:
                                    pass
                                if sen < 11 and sen > -1:
                                    if path[1] == 'phonearena':
                                        # sen = math.log(sen, 2.154) - 1
                                        if sen >= 8:
                                            positive_counter += 1
                                            sen = 2
                                        elif sen >= 4:
                                            sen = 1
                                            neutral_counter += 1
                                        else:
                                            sen = 0
                                            negative_counter += 1
                                    else:
                                        # nie ma wartosci neutrealnych dla labelek z amazona, imdb, yelp
                                        if sen == 1:
                                            sen += 1
                                            positive_counter_all += 1
                                        else:
                                            negative_counter_all += 1
                                    y.append(sen)
                                    docs.append(line)
                            except:
                                pass
        Y = np.array(y)
        print("Liczba pozytywnych, neutralnych i negatywnych zdan dla phonearena: ", positive_counter, ' ',
              neutral_counter, ' ', negative_counter)
        print("Liczba pozytywnych, neutralnych i negatywnych zdan dla wszystkich danych: ",
              positive_counter_all + positive_counter, ' ', neutral_counter, ' ',
              negative_counter_all + negative_counter)
        if return_stats:
            stats = [positive_counter_all + positive_counter, neutral_counter,
                     negative_counter_all + negative_counter]
            return stats, docs, Y
        else:
            return docs, Y

    def read_test_data(test_data_path):
        cameraData = []
        batteryData = []
        screenData = []
        cpuData = []
        fp = open(test_data_path)
        data = fp.read()

        # allData = tokenizer.tokenize(data)
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

        return allData, cameraData, batteryData, screenData, cpuData

    def transform_test_data(count_vect, allData, cameraData, batteryData, screenData, cpuData):
        """
        pobierac obiekt typu CountVectorizer, ktory posiada slownik zbudowany na danych uczacych i transformuje dane testowe
        :param count_vect:
        :return: ngrams
        """
        allDataX = ReadData.get_ngrams(allData, count_vect)
        cameraDataX = ReadData.get_ngrams(cameraData, count_vect)
        batteryDataX = ReadData.get_ngrams(batteryData, count_vect)
        screenDataX = ReadData.get_ngrams(screenData, count_vect)
        cpuDataX = ReadData.get_ngrams(cpuData, count_vect)

        return allDataX, cameraDataX, batteryDataX, screenDataX, cpuDataX

    # file: core_learning.py

    # returns pcaModel [type: PCA] (if not existed before pca_opt) and optimised array [type: sparse matrix]
    def pca_opt(sparseArray, pcaModel=None):
        array = sparseArray.toarray()
        if pcaModel is None:
            newSize = len(array[0]) / ReadData.svdOpt
            if len(array) < 3000:
                pca = PCA(n_components=newSize)
                Xtransformed = pca.fit_transform(array)
                Xtransformed = sparse.csr_matrix(Xtransformed)
            else:
                chunkSize = 20
                iter = int(len(array)/chunkSize)
                chunks = [array[i:i + chunkSize] for i in range(0, iter*chunkSize, chunkSize)]
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
                chunks = [array[i:i + chunkSize] for i in range(0, len(array), chunkSize)]

                Xtransformed = None
                for chunk in chunks:
                    Xchunk = pcaModel.transform(chunk)
                    if Xtransformed == None:
                        Xtransformed = Xchunk
                    else:
                        Xtransformed = np.vstack((Xtransformed, Xchunk))
                Xtransformed = sparse.csr_matrix(Xtransformed)

            return pcaModel, Xtransformed