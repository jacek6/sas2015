#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#@Author:PawełGrochowski
#

from KeyWords import KeyWords #@Local:../scraping

class DataStats():
    BROKEN = 'broken' # Sentences with multiple keywords or without rating.
    
    AMOUNT = 'amount' # Amount of sentences with single occurrence of keyword.
    RATING = 'rating' # Dictionary with GRADE:COUNT tuples.
    
    REASON = 'reason' # Dictionary with REASON:COUNT tuples describing broken sentences statistics.
    
    def __init__(self, content=[]):
        keyWords = KeyWords()
        
        self._mBroken = {DataStats.AMOUNT:0,DataStats.REASON:{}}
        self._mStats = dict()
        for key in keyWords.words:
            self._mStats[str(key).split('-', 1)[0]] = {
                    DataStats.AMOUNT:0,
                    DataStats.RATING:{}
            }
        
        for sentence in content:
            if len(sentence) != 2 or len(str(sentence[1])) == 0:
                reason = 'Missing rate.'
                self._mBroken[DataStats.AMOUNT] += 1
                if reason not in self._mBroken[DataStats.REASON]:
                    self._mBroken[DataStats.REASON][reason] = 1
                else:
                    self._mBroken[DataStats.REASON][reason] += 1
                continue
            if not isinstance(sentence[1], (int, long, float)):
                reason = 'Rating is not numeric.'
                self._mBroken[DataStats.AMOUNT] += 1
                if reason not in self._mBroken[DataStats.REASON]:
                    self._mBroken[DataStats.REASON][reason] = 1
                else:
                    self._mBroken[DataStats.REASON][reason] += 1
                continue
            sentenceKeyWords = keyWords.key_words_in_line(sentence[0])
            if len(sentenceKeyWords) > 1:
                reason = 'Sentence fit many keywords.'
                self._mBroken[DataStats.AMOUNT] += 1
                if reason not in self._mBroken[DataStats.REASON]:
                    self._mBroken[DataStats.REASON][reason] = 1
                else:
                    self._mBroken[DataStats.REASON][reason] += 1
                continue
            if len(sentenceKeyWords) == 0:
                reason = 'Sentence does not fit any keyword.'
                self._mBroken[DataStats.AMOUNT] += 1
                if reason not in self._mBroken[DataStats.REASON]:
                    self._mBroken[DataStats.REASON][reason] = 1
                else:
                    self._mBroken[DataStats.REASON][reason] += 1
                continue
            
            keyWord = str(sentenceKeyWords[0]).split('-', 1)[0]
            grade = sentence[1]
            
            self._mStats[keyWord][DataStats.AMOUNT] += 1
            if grade not in self._mStats[keyWord][DataStats.RATING]:
                self._mStats[keyWord][DataStats.RATING][grade] = 1
            else:
                self._mStats[keyWord][DataStats.RATING][grade] += 1
    ##
    
    def getBroken(self):
        return self._mBroken
    ##
    
    def getStats(self):
        return self._mStats
##
