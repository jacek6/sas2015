from dataRepresentation import *
import sys
import re

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
        result += [RatingDoc(div, url)]
        if printDoing: sys.stdout.write('.')
    return result

def isPhoneLink(link):
    p = re.compile(ur'\/phones\/([a-zA-Z\d\-]+\_id[\d]+)$')
    test_str = u"/phones/HTC-One-S9_id10040"
    found = re.search(p, link)
    if found:
        code = found.group(1)
        return code
    return False


def listPhonesLinksOnPage(url, printDoing=False):
    result = []
    r = urllib.urlopen(url).read()
    soup = BeautifulSoup(r, "html.parser")

    links = soup.findAll("a", {"class" : "atext"})
    links = soup.findAll("a")
    print links
    for link in links:
        #print dir(link)
        if 'href' not in link.attrs: continue
        target = link['href']
        if isPhoneLink(target):
            result.append(target)
    return result