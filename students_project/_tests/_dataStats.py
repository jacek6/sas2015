#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#@Author: Paweł Grochowski
#

import os
import json
import traceback
import collections
import numpy
import matplotlib.pyplot

from scraping.KeyWords import KeyWords #Run with 'PYTHONPATH=...' before!

def analyze(path):
    filePath = os.path.abspath(path)
    with open(filePath) as f:
        fileContent = f.read()
    
    try:
        dataJSON = json.loads(fileContent)
    except:
        raise Exception("Content of specified file must have valid JSON syntax!")
    
    RATING = "rating"
    RATING_FEATURES = "featuresRatings"
    NAME_FEATURE = "featureName"
    OTHER = "others"
    
    keyWords = KeyWords()
    keyWordsNames = [keyWord.get_aliases()[0] for keyWord in keyWords.words]
    
    ratingsElements = dict()
    ratingsFeatures = {keyWordName:dict() for keyWordName in keyWordsNames}
    ratingsFeatures[OTHER] = dict()
    for itemObject in dataJSON:
        rating = float(itemObject[RATING])
        if rating not in ratingsElements.keys():
            ratingsElements[rating] = 1
        else:
            ratingsElements[rating] += 1
        for feature in itemObject[RATING_FEATURES]:
            rating = float(feature[RATING])
            name = feature[NAME_FEATURE].lower()
            fitting = keyWords.key_words_in_line(name)
            if len(fitting) == 1:
                name = fitting[0].get_aliases()[0]
                if rating not in ratingsFeatures[name].keys():
                    ratingsFeatures[name][rating] = 1
                else:
                    ratingsFeatures[name][rating] += 1
            else:
                if rating not in ratingsFeatures[OTHER].keys():
                    ratingsFeatures[OTHER][rating] = 1
                else:
                    ratingsFeatures[OTHER][rating] += 1
    ratingsElements = collections.OrderedDict(sorted(ratingsElements.items()))
    ratingsFeatures = {item[0]:collections.OrderedDict(sorted(item[1].items())) for item in ratingsFeatures.items()}
    
#     DISTRIBUTIONS = [        
#         scipy.stats.ksone,scipy.stats.kstwobign,scipy.stats.norm,scipy.stats.alpha,scipy.stats.anglit,scipy.stats.arcsine,
#         scipy.stats.beta,scipy.stats.betaprime,scipy.stats.bradford,scipy.stats.burr,scipy.stats.fisk,scipy.stats.cauchy,
#         scipy.stats.chi,scipy.stats.chi2,scipy.stats.cosine,scipy.stats.dgamma,scipy.stats.dweibull,scipy.stats.erlang,
#         scipy.stats.expon,scipy.stats.exponweib,scipy.stats.exponpow,scipy.stats.fatiguelife,scipy.stats.foldcauchy,
#         scipy.stats.f,scipy.stats.foldnorm,scipy.stats.frechet_r,scipy.stats.weibull_min,scipy.stats.frechet_l,
#         scipy.stats.weibull_max,scipy.stats.genlogistic,scipy.stats.genpareto,scipy.stats.genexpon,scipy.stats.genextreme,
#         scipy.stats.gamma,scipy.stats.gengamma,scipy.stats.genhalflogistic,scipy.stats.gompertz,scipy.stats.gumbel_r,
#         scipy.stats.gumbel_l,scipy.stats.halfcauchy,scipy.stats.halflogistic,scipy.stats.halfnorm,scipy.stats.hypsecant,
#         scipy.stats.gausshyper,scipy.stats.invgamma,scipy.stats.invgauss,scipy.stats.invweibull,
#         scipy.stats.johnsonsb,scipy.stats.johnsonsu,scipy.stats.laplace,scipy.stats.levy,scipy.stats.levy_l,
#         scipy.stats.levy_stable,scipy.stats.logistic,scipy.stats.loggamma,scipy.stats.loglaplace,scipy.stats.lognorm,
#         scipy.stats.gilbrat,scipy.stats.maxwell,scipy.stats.mielke,scipy.stats.nakagami,scipy.stats.ncx2,scipy.stats.ncf,scipy.stats.t,
#         scipy.stats.nct,scipy.stats.pareto,scipy.stats.lomax,scipy.stats.pearson3,scipy.stats.powerlaw,scipy.stats.powerlognorm,
#         scipy.stats.powernorm,scipy.stats.rdist,scipy.stats.rayleigh,scipy.stats.reciprocal,scipy.stats.rice,
#         scipy.stats.recipinvgauss,scipy.stats.semicircular,scipy.stats.triang,scipy.stats.truncexpon,
#         scipy.stats.truncnorm,scipy.stats.tukeylambda,scipy.stats.uniform,scipy.stats.vonmises,scipy.stats.vonmises_line,
#         scipy.stats.wald,scipy.stats.wrapcauchy
#     ]
    
    rCount = ratingsElements.values()
    rRate = ratingsElements.keys()
    
    barWidth=0.8
    xAxis=numpy.arange(len(rCount))
    matplotlib.pyplot.figure("RATINGS:General")
    matplotlib.pyplot.bar(xAxis,rCount,barWidth)
    matplotlib.pyplot.xticks(xAxis+barWidth*0.5,rRate,rotation=90)
    matplotlib.pyplot.title("Distribution of general ratings amounts:")
    matplotlib.pyplot.xlabel("rating")
    matplotlib.pyplot.ylabel("amount")
    
#     with warnings.catch_warnings():
#         warnings.filterwarnings('ignore')
#          
#         for distribution in DISTRIBUTIONS:
#             params = distribution.fit(rCount)
#             arg = params[:-2]
#             loc = params[-2]
#             scale = params[-1]
#             xmin, xmax = matplotlib.pyplot.xlim()
#             x = numpy.linspace(xmin, xmax, 100)
#             p = distribution.pdf(x, loc=loc, scale=scale, *arg)
#             matplotlib.pyplot.plot(x, p, 'r-', linewidth=2)
    
    for feature in ratingsFeatures.items():
        rCount = feature[1].values()
        rRate = feature[1].keys()
        
        barWidth=0.8
        xAxis=numpy.arange(len(rCount))
        matplotlib.pyplot.figure("RATINGS:%s" % str(feature[0]))
        matplotlib.pyplot.bar(xAxis,rCount,barWidth)
        matplotlib.pyplot.xticks(xAxis+barWidth*0.5,rRate,rotation=90)
        matplotlib.pyplot.title("Distribution of '%s' ratings amounts:" % str(feature[0]))
        matplotlib.pyplot.xlabel("rating")
        matplotlib.pyplot.ylabel("amount")
        
#         with warnings.catch_warnings():
#             warnings.filterwarnings('ignore')
#              
#             for distribution in DISTRIBUTIONS:
#                 params = distribution.fit(rCount)
#                 arg = params[:-2]
#                 loc = params[-2]
#                 scale = params[-1]
#                 xmin, xmax = matplotlib.pyplot.xlim()
#                 x = numpy.linspace(xmin, xmax, 100)
#                 p = distribution.pdf(x, loc=loc, scale=scale, *arg)
#                 matplotlib.pyplot.plot(x, p, 'r-', linewidth=2)
    
    matplotlib.pyplot.show()
##

if __name__ == "__main__":
    try:
        analyze("../scraping/data/json_data.txt") # WTF: Extension should be '.json'!
    except KeyboardInterrupt:
        print("Interrupted by user!")
    except Exception as e:
        print("ERROR: '%s'" % str(e))
        traceback.print_exc()
##
