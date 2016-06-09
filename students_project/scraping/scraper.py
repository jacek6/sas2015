from phoneScraper import scrapPhone
from KeyWords import *
import json
import threading
import os.path

SENTENCES_DATA_FILENAME_PREFIX = "sentences_data"

FILENAME_SUFIX = '.txt'

SIMPLE_DATA_FILENAME_PREFIX = "simple_data"
JSON_FILENAME_PREFIX = "json_data"

phoneCodesFile = 'phonesCodes.txt'
printDoing = True

keyWords = KeyWords()


class SimpleDataWriter:
    base_filename = SIMPLE_DATA_FILENAME_PREFIX
    #filename = "simple_data.txt"

    def __init__(self, base_filename=SIMPLE_DATA_FILENAME_PREFIX):
        self.base_filename = base_filename
        self.use_filename_sufix()
        return

    def use_filename_sufix(self, sufix=''):
        self.filename = self.base_filename + sufix + FILENAME_SUFIX
        return

    def write_to_file(self, ratingDoc, file):
        for p in ratingDoc.ps:
            file.write(p.replace('\n', ' '))
        file.write('\t')
        file.write(ratingDoc.rating)
        file.write('\n')
        return

    def docs_to_file(self, docs, filename):
        with open(filename, 'ab') as file:
            for doc in docs:
                self.write_to_file(doc, file)
        return

    def feed_docs(self, docs):
        self.docs_to_file(docs, 'data/%s' % self.filename)
        return


class SentencesWriter(SimpleDataWriter):
    base_filename = SENTENCES_DATA_FILENAME_PREFIX
    #filename = "sentences_data.txt"

    def __init__(self, base_filename=SENTENCES_DATA_FILENAME_PREFIX):
        self.base_filename = base_filename
        self.use_filename_sufix()
        return

    def write_to_file(self, ratingDoc, file):
        for p in ratingDoc.ps:
            for sentence in p.split('.'):
                for subsentence in sentence.split(','):
                    self.write_subsentence(file, ratingDoc, subsentence.strip())

    def write_subsentence(self, file, ratingDoc, subsentence):
        if not subsentence: return
        words = keyWords.key_words_in_line(subsentence.lower())
        if len(words) != 1: return
        word = words[0]
        feature = self.find_feature(ratingDoc, word)
        if not feature: return
        #file.write(ratingDoc.url)
        #file.write('\t')
        file.write(subsentence)
        file.write('\t')
        file.write(ratingDoc.date)
        file.write('\t')
        file.write(feature.ratingGainPoints)
        file.write('\n')

    def find_feature(self, ratingDoc, word):
        for feature in ratingDoc.featuresRatings:
            if word.is_there(feature.featureName.lower()):
                return feature
        return None

class JsonWriter(SimpleDataWriter):
    def __init__(self, base_filename=JSON_FILENAME_PREFIX):
        SimpleDataWriter.__init__(self, base_filename)

    def write_to_file(self, ratingDoc, file):
        file.write(json.dumps(ratingDoc.dict_me()))
        file.write(',')
        file.write('\n')
        return


#writers = [SimpleDataWriter()]
#writers2 = [SimpleDataWriter(), SentencesWriter()]


class ScrapingPhonesThread(threading.Thread):
    def __init__(self, phoneCodes, writers=[SimpleDataWriter()], printDoing=True, filename_sufix=''):
        threading.Thread.__init__(self)
        self.phoneCodes = phoneCodes
        self.writers = writers
        self.printDoing = printDoing
        self.filename_sufix = filename_sufix
        return

    def run(self):
        print "Run thread with %d chunks and sufix: %s" % (len(self.phoneCodes), self.filename_sufix)
        for phoneCode in self.phoneCodes:
            if self.printDoing: print phoneCode
            docs = scrapPhone(phoneCode, self.printDoing)
            for writer in self.writers:
                writer.use_filename_sufix(self.filename_sufix)
                writer.feed_docs(docs)
        return


def feed_writers(phoneCodes, writers=[SimpleDataWriter()], printDoing=True, filename_sufix=''):
    for phoneCode in phoneCodes:
        docs = scrapPhone(phoneCode, printDoing)
        for writer in writers:
            writer.use_filename_sufix(filename_sufix)
            writer.feed_docs(docs)
    return


def gen_phone_codes_from_json(phoneCodesFile='phones_links.json'):
    result = []
    with open(phoneCodesFile, "r") as file_contents:
        json_obj = json.load(file_contents)
        for o2 in json_obj:
            link = o2['phoneLink']  # /phones/HTC-One-S9_id10040
            phoneCode = link[8:]  # HTC-One-S9_id10040
            # print phoneCode
            result.append(phoneCode)
    return result

def mergeFiles(prefix, sufix, from_, to, json_array=False):
    with open(prefix + sufix, 'wb') as target_file:
        if json_array:
            target_file.write('[\n')
        for i in range(from_, to):
            reading_filename = prefix + str(i) + sufix
            if not os.path.exists(reading_filename): continue
            append_file(reading_filename, target_file)
        if json_array:
            target_file.write('\n{}\n]\n')
    return


def append_file(reading_filename, target_file):
    with open(reading_filename, 'rb') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                target_file.write(line)
                target_file.write('\n')


def chunks(l, n):
    """

    :param l: array to chunk
    :param n: size of one of chunks
    :return: array of chunks (with is arraies)
    """
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]

def removeDublets(array):
    array = sorted(array)
    prev_val = None
    result = []
    for val in array:
        if val == prev_val: continue
        result.append(val)
        prev_val = val
    return result

'''
gen_txt_files(phoneCodesFile, writers2, printDoing)
print ''
print ''
print "everything done fine"
'''

numThreads = 800

#mergeFiles('data/%s' % SIMPLE_DATA_FILENAME_PREFIX, FILENAME_SUFIX, 1, numThreads)
#mergeFiles('data/%s' % SENTENCES_DATA_FILENAME_PREFIX, FILENAME_SUFIX, 1, numThreads)
mergeFiles('data/%s' % JSON_FILENAME_PREFIX, FILENAME_SUFIX, 1, numThreads, True)
exit(0)

phoneCodes = gen_phone_codes_from_json()
phoneCodes = removeDublets(phoneCodes)

phoneCodesChunks = chunks(phoneCodes, len(phoneCodes) / numThreads)
sufix_counter = 1
threads = []
for chunk in phoneCodesChunks:
    #thread_writers = [SimpleDataWriter(), SentencesWriter()]
    thread_writers = [JsonWriter()]
    thread = ScrapingPhonesThread(phoneCodes=chunk, filename_sufix=str(sufix_counter), writers=thread_writers)
    sufix_counter += 1
    thread.start()
    threads.append(thread)

joined_threads = 0
for thread in threads:
    joined_threads += 1
    thread.join()
    print 'joined thread %d ' % joined_threads

print "JOINED ALL THREADS"

#mergeFiles('data/%s' % SIMPLE_DATA_FILENAME_PREFIX, FILENAME_SUFIX, 1, numThreads)
#mergeFiles('data/%s' % SENTENCES_DATA_FILENAME_PREFIX, FILENAME_SUFIX, 1, numThreads)
mergeFiles('data/%s' % JSON_FILENAME_PREFIX, FILENAME_SUFIX, 1, numThreads, True)

# t1 = ScrapingPhonesThread(phoneCodes = phoneCodesChunks[0], filename_sufix=str(1))
# t1.start()
