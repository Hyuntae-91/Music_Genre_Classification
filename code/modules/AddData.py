#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def AddData(temp_freq,freqN,n):
    # Sort in descending order
    temp_freq = temp_freq.sort_values(by=['freq'],axis=0,ascending=False)
    
    # Insert Maximum value data
    freqN.ix[n] = temp_freq.ix[0]
    return freqN
