#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def TableIloc(table):
    # Remove data from table except for first to 12th candidate
    table = table.iloc[:12]

    return table

