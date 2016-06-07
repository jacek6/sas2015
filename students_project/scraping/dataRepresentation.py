from bs4 import BeautifulSoup
import urllib
import re

#import scrapy

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
    """
    See doc: https://github.com/jacek6/sas2015/wiki/Docs---Scraping
    """
    url = ''
    title = ''
    date = ''
    rating = ''
    pros = ['']
    cons =['']
    ps = [''] # akapity
    featuresRatings = [FeatureRating()]
    ratingDocUsable = RatingDocUsable()

    def dict_me(self):
        """
        Na podstawie tego jest generowany json
        """
        dict = self.__dict__
        dict['featuresRatings'] = [fr.__dict__ for fr in self.featuresRatings]
        dict['ratingDocUsable'] = self.ratingDocUsable.__dict__
        return  dict

    def __init__(self, div=None, url = ''):
        self.url = url
        if not div: return
        self.title = self.toText(div.find("h3", {"class" : "clear"}))
        self.date = self.toText(div.find("span", {"class" : "s_date"}))
        self.rating = self.toText(div.find("span", {"class" : "s_rating_overal"}))
        self.pros = self.ulToLis(div.find("ul", {"class" : "s_pros_list"}))
        self.cons = self.ulToLis(div.find("ul", {"class" : "s_cons_list"}))
        self.ps = [p.getText().encode('utf-8') for p in div.findAll("p", {"class": "s_desc"})]

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
