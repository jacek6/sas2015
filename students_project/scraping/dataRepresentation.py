from bs4 import BeautifulSoup
import urllib
import re

class RatingDocUsable:

    factor = 0.0
    usableSum = 0
    usableTotal = 0
    rawText = ''

    def __init__(self, rawText=''):
        if not rawText: return
        # rawText = str(rawText) # uncomment if given rawText is not 'real' string
        self.rawText = rawText
        regex = '[\h]*([\d]+)\ out\ of\ ([\d]+)\ people\ found'
        m = re.search(regex, rawText)
        if not m:
            #print "Regex fail: >>>%s<<<<" % (rawText)
            return
        self.usableSum = int(m.group(1))
        self.usableTotal = int(m.group(2))
        self.factor = float(self.usableSum) / float(self.usableTotal)

    def __str__(self):
        return "Usable %.2f (%d/%d)" % (self.factor, self.usableSum, self.usableTotal)

    def __repr__(self):
        return self.__str__()


class FeatureRating:

    def __init__(self, div=None):
        if not div: return
        self.featureName =  div.find("strong", {"class" : "rating_name"}).getText()
        self.rawRating = div.find("span", {"class" : "s_total_votes"}).getText()
        self.ratingGainPoints, self.ratingTotatlPoints = self.rawRating.split('/')
        self.comment =  div.find("span", {"class" : "rating-comment"}).getText()
        self.rating = float(self.ratingGainPoints) / float(self.ratingTotatlPoints)

    def __str__(self):
        return "%s (%.2f): %s" % (self.featureName, self.rating, self.comment)

    def __repr__(self):
        return self.__str__()

class RatingDoc:
    title = ''
    rating = ''
    pros = ['']
    cons =['']
    ps = [''] # akapity
    featuresRatings = [FeatureRating()]
    ratingDocUsable = RatingDocUsable()


    def __init__(self, div=None):
        if not div: return
        self.title = self.toText(div.find("h3", {"class" : "clear"}))
        self.mark = self.toText(div.find("span", {"class" : "s_rating_overal"}))
        self.pros = self.ulToLis(div.find("ul", {"class" : "s_pros_list"}))
        self.cons = self.ulToLis(div.find("ul", {"class" : "s_cons_list"}))
        self.ps = [p.getText() for p in div.findAll("p")]

        self.featuresRatings = []
        for featureDiv in div.findAll("div", { "class" : "s_category_overal left" }):
            self.featuresRatings.append(FeatureRating(featureDiv))
        self.ratingDocUsable = RatingDocUsable(div.find("p", {"class" : "s_f_12 s_mr_10 gray_9 left"}).getText())

    def toText(self, obj, noneVal=None):
        if obj:
            return obj.getText()
        return noneVal

    def ulToLis(self, ul, noneVal=[]):
        if ul:
            return [li.getText() for li in ul.findAll("li")]
        return noneVal
