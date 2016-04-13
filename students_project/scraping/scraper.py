from phoneScraper import scrapPhone

phoneCodesFile = 'phonesCodes.txt'
printDoing=True


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
        with open(filename, 'a') as file:
            for doc in docs:
                self.write_to_file(doc, file)
        return

    def feed_docs(self, docs):
        self.docs_to_file(docs, 'data/%s' % self.filename)
        return


writers = [SimpleDataWriter()]

def gen_txt_files(phoneCodesFile = 'phonesCodes.txt', writers=[SimpleDataWriter()], printDoing=True):
    with open(phoneCodesFile, "r") as f:
        for line in f:
            phoneCode = line.rstrip('\n')
            print ''
            print phoneCode
            docs = scrapPhone(phoneCode, printDoing)
            for writer in writers:
                writer.feed_docs(docs)

gen_txt_files(phoneCodesFile, writers, printDoing)