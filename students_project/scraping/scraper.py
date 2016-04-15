from phoneScraper import scrapPhone
from KeyWords import *

phoneCodesFile = 'phonesCodes.txt'
printDoing=True

keyWords = KeyWords()

class SimpleDataWriter:
    filename = "simple_data.txt"

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
    filename = "sentences_data.txt"

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

        file.write(subsentence)
        file.write('\t')
        file.write(feature.ratingGainPoints)
        file.write('\n')

    def find_feature(self, ratingDoc, word):
        for feature in ratingDoc.featuresRatings:
            if word.is_there(feature.featureName.lower()):
                return feature
        return None


writers = [SimpleDataWriter()]
writers2 = [SimpleDataWriter(), SentencesWriter()]

def gen_txt_files(phoneCodesFile = 'phonesCodes.txt', writers=[SimpleDataWriter()], printDoing=True):
    with open(phoneCodesFile, "r") as f:
        for line in f:
            phoneCode = line.rstrip('\n')
            print ''
            print phoneCode
            docs = scrapPhone(phoneCode, printDoing)
            for writer in writers:
                writer.feed_docs(docs)

gen_txt_files(phoneCodesFile, writers2, printDoing)
print ''
print ''
print "everything done fine"