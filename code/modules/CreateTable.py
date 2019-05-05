#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Create Table to save music data

from pandas import Series, DataFrame

def CreateTable():
    data={}
    indexing = list(range(0,32))
    return DataFrame(data, columns =['freq','time'], index = indexing)

