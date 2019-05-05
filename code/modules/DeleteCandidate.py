#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def DeleteCandidate(table,temp):
    # If the time value were less then 32, delete the row for that label from the table
    num = len(temp)-1
    while(num>-1):
        table.ix[temp[num]]=[np.NaN,np.NaN]
        num -= 1
        table = tableRename(table)

    return table

