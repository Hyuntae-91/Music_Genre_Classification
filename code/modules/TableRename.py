#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def TableRename(table):
    # Rename label of row to 0 ~ 11
    table = table.rename(index={table.index[0]:0,table.index[1]:1,table.index[2]:2,table.index[3]:3,
                                table.index[4]:4,table.index[5]:5,table.index[6]:6,table.index[7]:7,
                                table.index[8]:8,table.index[9]:9,table.index[10]:10,table.index[11]:11,})
    return table
