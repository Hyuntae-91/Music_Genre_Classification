#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# When sort the candidate of frequency, using this function

def SortCandidate(table,ascending):
    if(ascending == False):
        # Sort frequencies in descending order
        table = table.sort_values(by=['freq'],axis=0,ascending=False)
    elif(ascending == True):
        # Sort time value in ascending order
        table = table.sort_values(by=['time'],axis=0,ascending=True)

    return table
