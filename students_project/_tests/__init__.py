#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#@Author:PawełGrochowski
#

import os

__all__ = []

___cwd = os.getcwd()
___files = [f for f in os.listdir(___cwd) if os.path.isfile(os.path.join(___cwd, f))]
for f in ___files:
    if not f[0].isupper(): continue
    if f[-3:] == '.py': __all__.append(f)
