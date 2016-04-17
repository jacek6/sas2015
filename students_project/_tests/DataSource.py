#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#@Author:PawełGrochowski
#

import pandas
import numpy

from DataStats import DataStats

class DataSource():
    NAME = 'name'
    PATH = 'path'
    TYPE = 'type'
    
    REGULAR = 'regular'
    BROKEN = 'broken'
    
    def __init__(self, src=None):
        self.__mClassName = str(type(self))
        
        if not isinstance(src, dict): raise Exception("%s class constructor parameter must be a dictionary!" % self.__mClassName)
        
        if DataSource.NAME and DataSource.PATH and DataSource.TYPE not in src:
            raise Exception("Dictionary passed to %s class constructor must consist of three keys: '%s', '%s', '%s'"
                            % (self.__mClassName, DataSource.NAME, DataSource.PATH, DataSource.TYPE))
        
        self._mName = src[DataSource.NAME]
        self._mType = src[DataSource.TYPE]
        self._mPath = src[DataSource.PATH]
    ##
    
    def __readTXT(self):
        with open(self._mPath) as f: return [line.replace('\n', '').rsplit('\t', 1) for line in f.readlines()]
    ##
    
    def __readCSV(self):
        return numpy.fliplr(pandas.read_csv(self._mPath).as_matrix()[1:,1:]).tolist()
    ##
    
    def getStats(self):
        content = self.getContent()
        for sentence in content:
            try: sentence[1] = int(sentence[1])
            except: pass
        
        dStats = DataStats(content)
        
        return {
            DataSource.REGULAR : dStats.getStats(),
            DataSource.BROKEN : dStats.getBroken()
        }
    ##
    
    def getContent(self):
        if self._mType == '.csv':
            return self.__readCSV()
        elif self._mType == '.txt':
            return self.__readTXT()
        else:
            raise Exception("Unsupported type!")
##
