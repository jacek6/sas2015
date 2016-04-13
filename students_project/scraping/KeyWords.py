class KeyWord:

    def __init__(self, aliases):
        self.aliases = aliases
        pass

    def get_aliases(self):
        return self.aliases

    def is_there(self, line):
        for alias in self.aliases:
            if alias in line: return True
        return False

    def __str__(self):
        return '-'.join(self.aliases)

    def __repr__(self):
        return self.__str__()


class KeyWords:

    def __init__(self, filename='../slowa - klucz.txt'):
        self.words = []
        with open(filename) as file:
            for line in file:
                line = line.strip()
                if not line: return
                aliases = [s.strip() for s in line.split(',')]
                self.words.append(KeyWord(aliases))

    def key_words_in_line(self, line):
        return [w for w in self.words if w.is_there(line)]



#words = KeyWords()
#print words.key_words_in_line("camera colour sd sdsdkjdjkdjkd")