from dataRepresentation import *
import sys

def scrapPhone(phoneCode, printDoing=False):
    result = []
    page = 1
    while 1:
        urlPattern = 'http://www.phonearena.com/phones/%s/reviews/sort/default/page/%d'
        url = urlPattern % (phoneCode, page)
        try:
            if printDoing: sys.stdout.write('|')
            adding = scrapPhonePage(url, printDoing)
            if not adding: break
            result += adding
            page += 1
        except IOError:
            break
    return result

def scrapPhonePage(url, printDoing=False):
    result = []
    r = urllib.urlopen(url).read()
    soup = BeautifulSoup(r, "html.parser")

    mydivs = soup.findAll("div", { "class" : "s_user_review s_post s_block_1 s_block_1_s3 clearfix" })
    for div in mydivs:
        if printDoing: sys.stdout.write(' ')
        result += [RatingDoc(div)]
        if printDoing: sys.stdout.write('.')
    return result