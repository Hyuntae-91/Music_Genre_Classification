#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pydub import AudioSegment
import os

def Load_Music_Data(DIR, File_name):
    if not os.path.isfile(DIR + "/%s.wav" % File_name):
        PARENT_DIR = os.path.dirname(DIR)
        Loaded_mp3_file = AudioSegment.from_mp3(PARENT_DIR + "/%s.mp3" % File_name)
        Loaded_mp3_file.export(DIR + "/%s.wav" % File_name, format="wav")

    Music_Data = DIR + "/" + str(File_name) + ".wav"
    return Music_Data
