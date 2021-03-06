#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import librosa
from pandas import Series, DataFrame
import numpy as np
import pandas as pd
from .AddData import AddData
from .ChoiceCandidate import ChoiceCandidate
from .CreateTable import CreateTable
from .SortCandidate import SortCandidate
from .TableIloc import TableIloc
from .TableRename import TableRename


def Climax(sample, y, sr):
    # Create Table for save music frequency samples
    freqTable = []
    for i in range(7):
        freqTable.append(CreateTable()) 
    temp_freq = CreateTable()

    # Store the largest value in the 0th to 6th frequency bands in 32 sections 
    for i in range(0,7):
        n=0
        for j in range(0,1024):
            temp_freq.ix[j%32] = [sample[i][j],j]
            if(j != 0 and j%32 == 0):
                freqTable[i] = AddData(temp_freq,freqTable[i],n)
                n += 1
            elif(j==1023):
                freqTable[i] = AddData(temp_freq,freqTable[i],n)
                n += 1

    
    for i in range(7):
        freqTable[i] = SortCandidate(freqTable[i],False)
        freqTable[i] = TableIloc(freqTable[i])
        freqTable[i] = TableRename(freqTable[i])
        freqTable[i] = ChoiceCandidate(freqTable[i])
        freqTable[i] = SortCandidate(freqTable[i],True)
        freqTable[i] = TableRename(freqTable[i])


    # The candidates left by each frequency are compared with candidates of different frequencies.
    # Find the value with time difference of +10, -10 or less and count.
    # Then calculate the position value average with the result of 4 or more.
    time_list=[]
    for i in range(4):
        for j in range(12):
            count = 1
            temp = [freqTable[i].ix[j].time]
            for k in range(i+1,7):
                for m in range(12):
                    if(freqTable[i].ix[j].time <= freqTable[k].ix[m].time + 10 and freqTable[i].ix[j].time >= freqTable[k].ix[m].time - 10):
                        count += 1
                        temp.append(freqTable[k].ix[m].time)
            if(count>=4):
                time_list.append(np.mean(temp)) # Calculate average of temp value and add it on time_list

    # Sort calculated averages in ascending order
    time_list.sort()

    time_interval = []
    # Find the time interval between each candidates
    for i in range(len(time_list)-1):            
        time_interval.append(time_list[i+1] - time_list[i])

    if(len(time_interval)<=1):
        time_interval.append(861)
    # Sorting the time intervals in ascending order
    time_interval.sort()   

    if(len(time_interval)>2):
        # Pop the largest value of the time interval 
        time_interval.pop(len(time_interval)-1)

    if(len(time_interval)>2):
        # Pop the smallest value of the time interval
        time_interval.pop(0)

    # Length of a bar
    bar_length = np.mean(time_interval)*2*256
    # Start point of a bar
    # bar_start = (max(time_interval)-(np.mean(time_interval)/2))*256
    bar_start = bar_length/4 
    # Length of Climax
    climax_length = bar_length*8


    max_wave = max(y) # Largest waveform in a song
    music_length = y.shape # total length of a song
    bar_num = int(music_length/bar_length)+1 # Number of bar

    music_divide = [] # Save music by each section 
    n=0
    for i in range(0,bar_num):
        music_divide.append(y[n:int(bar_length)*(i+1)])
        n = n+int(bar_length)

    # Count and store more than 90% of the largest waveform of the song at each node
    save_count = []
    for i in range(0,bar_num):
        count = 0
        for j in range(0,len(music_divide[i])):
            if(music_divide[i][j] >= max_wave*0.9):
                count += 1
        save_count.append(count)

    # The most bar which has more than 90% of largest waveform is climax
    bar_climax= save_count.index(max(save_count))
    climax = bar_climax*bar_length

    music_start = climax/sr 
    music_duration = (climax/sr + climax_length/sr) - climax/sr 
    realbar_start = bar_start/sr

    return music_start, music_duration, realbar_start
