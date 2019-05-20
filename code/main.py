#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import joblib
import os
from modules.Classifier import Classifier
from modules.Load_Music_Data import Load_Music_Data

MUSIC_DATA_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/Music/wav"

Feature = [[0]*256]

File_Name = 'test_data'

Music_data = Load_Music_Data(MUSIC_DATA_DIR, File_Name)

predict = Classifier(Music_data)

print(predict)

