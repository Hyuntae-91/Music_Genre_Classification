#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .DeleteCandidate import DeleteCandidate

# Discards a value whose time position value is less than 32
def ChoiceCandidate(table):
    for i in range(0, len(table.index)):
        temp = [] # List which save the label of value with time position value less than 32
        for j in range(i+1,len(table.index)):
            if(table.ix[i].time < table.ix[j].time+32 and table.ix[i].time > table.ix[j].time-32):
                temp.append(j)
        table = DeleteCandidate(table,temp)

    return table
