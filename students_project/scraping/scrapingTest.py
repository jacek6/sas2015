import json

from dataRepresentation import *

# ten przyklad zczytuje wszystkie opinie z podanej strony:
r = urllib.urlopen('http://www.phonearena.com/phones/Samsung-Galaxy-S5_id8202/reviews').read()
soup = BeautifulSoup(r, "html.parser")

mydivs = soup.findAll("div", { "class" : "s_user_review s_post s_block_1 s_block_1_s3 clearfix" })
for div in mydivs:
    doc1 = RatingDoc(div)
    #print doc1.title
    print 'found json '
    print json.dumps(doc1.dict_me())