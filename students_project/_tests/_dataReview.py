#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#@Author:PawełGrochowski
#

import traceback
import os
import numpy
import matplotlib.pyplot
import pprint
import string

from DataSource import DataSource
from DataStats import DataStats

if __name__ == "__main__":
    try:
        srcs = [
                {
                    DataSource.NAME:'Amazon',
                    DataSource.PATH:'../amazon_cells_labelled.txt'
                },
                {
                    DataSource.NAME:'PhoneArena',
                    DataSource.PATH:'../scraping/data/sentences_data.txt'
                },
                {
                    DataSource.NAME:'Twitter',
                    DataSource.PATH:'../../data/SemEval-2014.csv'
                }
        ]
        
        srcs = [{DataSource.NAME:src[DataSource.NAME],
                DataSource.PATH:os.path.abspath(src[DataSource.PATH]),
                DataSource.TYPE:os.path.splitext(src[DataSource.PATH])[1]} for src in srcs]
        
        for src in srcs:
            print("Name: '%s'\nType: '%s'" % (src[DataSource.NAME], src[DataSource.TYPE]))
            dSrc = DataSource(src)
            srcStats = dSrc.getStats()
            print("\nUsable data:")
            pprint.pprint(srcStats[DataSource.REGULAR])
            print("\nUnusable data:")
            pprint.pprint(srcStats[DataSource.BROKEN])
            print("-----------------")
        
        print("Finished!")
        
        for src in srcs:
            srcName = src[DataSource.NAME]
            dSrc = DataSource(src)
            srcStats = dSrc.getStats()
            kws,ocs = [],[]
            for word in srcStats[DataSource.REGULAR].items():
                ocs.append(word[1][DataStats.AMOUNT])
                kws.append(word[0]+'\n['+str(ocs[-1])+']')
            kwb,ocb = [],[]
            desc,lbls = [],[]
            alpha = list(string.ascii_uppercase)
            idx = 0
            for reason in srcStats[DataSource.BROKEN][DataStats.REASON].items():
                ocb.append(reason[1])
                kwb.append(alpha[idx]+'\n['+str(ocb[-1])+']')
                lbls.append(alpha[idx])
                desc.append(reason[0])
                idx += 1
            
            lDesc = [matplotlib.lines.Line2D([0], [0], linestyle='none', mfc='black',
                        mec='none', marker=r'$\mathregular{{{}}}$'.format(lbl))for lbl in lbls]
            
            barWidth=0.8
            matplotlib.pyplot.figure(srcName)
            matplotlib.pyplot.suptitle('Data origin: \'%s\'' % srcName)
            
            xAxis=numpy.arange(len(ocs))
            matplotlib.pyplot.subplot(121)
            matplotlib.pyplot.bar(xAxis,ocs,barWidth)
            matplotlib.pyplot.xticks(xAxis+barWidth*0.5,kws)
            matplotlib.pyplot.title('Histogram of usable data:')
            matplotlib.pyplot.xlabel('keyword [name : count]')
            matplotlib.pyplot.ylabel('sentences [count]')
            
            xAxis=numpy.arange(len(ocb))
            matplotlib.pyplot.subplot(122)
            matplotlib.pyplot.bar(xAxis,ocb,barWidth)
            matplotlib.pyplot.xticks(xAxis+barWidth*0.5,kwb)
            matplotlib.pyplot.legend(lDesc, desc, numpoints=1, markerscale=2, bbox_to_anchor=(1.3, 1.02), fontsize=10)
            matplotlib.pyplot.title('Histogram of unusable data:')
            matplotlib.pyplot.xlabel('reason [label : count]')
            
            kWordsCount = len(srcStats[DataSource.REGULAR].items())
            idx = 0
            barWidth=0.8
            labeled = False
            matplotlib.pyplot.figure(srcName+':RATES')
            matplotlib.pyplot.suptitle('Histograms of keywords ratings: [Data origin: \'%s\']' % srcName)
            for keyWord in srcStats[DataSource.REGULAR].items():
                kwRate = []
                rCount = []
                for rating in srcStats[DataSource.REGULAR][keyWord[0]][DataStats.RATING].items():
                    rCount.append(rating[1])
                    kwRate.append(str(rating[0])+'\n['+str(rCount[-1])+']')
                
                idx += 1
                xAxis=numpy.arange(len(rCount))
                matplotlib.pyplot.subplot(int('1'+str(kWordsCount)+str(idx)))
                matplotlib.pyplot.bar(xAxis,rCount,barWidth)
                matplotlib.pyplot.xticks(xAxis+barWidth*0.5,kwRate)
                matplotlib.pyplot.title('\'%s\':' % keyWord[0])
                matplotlib.pyplot.xlabel('rate [value : count]')
                if not labeled: matplotlib.pyplot.ylabel('count [count]')
        
        matplotlib.pyplot.show()
        
    except KeyboardInterrupt:
        print("Interrupted by user!")
    except:
        traceback.print_exc()
##
