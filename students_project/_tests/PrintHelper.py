#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#@Author:PawełGrochowski
#

from __builtin__ import str

import pprint

class PrintHelper():
    """
    Base class for on/off print support.
    """
    def __init__(self, doPrint=False):
        self._mDoPrint = doPrint
    def _print(self, what):
        if isinstance(what, str): print(what)
        else: pprint.pprint(what)
##
