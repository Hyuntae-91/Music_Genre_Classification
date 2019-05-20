#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import joblib
import os
from modules import Extract_Feature
from modules import Load_Music_Data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

TRAINED_DATA_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/Trained_Data_Set"
MUSIC_DATA_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/Music/wav"

Num_of_Music = 200
Classes = 10
Class_Range = Num_of_Music / Classes
    
Features = [[0]*254 for i in range(Num_of_Music)]

# 0 : classic // 1 : Hip-Hop // 2 : Reggae // 3: blues // 4 : alternative rock
# 5 : Heavy metal // 6 : House // 7 : jazz // 8 : folk // 9 : Country
for i in range(0, Num_of_Music):
    Music_data = Load_Music_Data(MUSIC_DATA_DIR, Num_of_Music)
    Features[i] = Extract_Feature(Music_data)

X = data
y = []

for i in range (0, Num_of_Music):
    if(i < Class_Range): # classic
        y.append(0)
    elif(i >= Class_Range and i < Class_Range * 2): # Hip-Hop
        y.append(1)
    elif(i >= Class_Range * 2 and i < Class_Range * 3): # Raggae
        y.append(2)
    elif(i >= Class_Range * 3 and i < Class_Range * 4): # blues
        y.append(3)
    elif(i >= Class_Range * 4 and i < Class_Range * 5): # alternative rock
        y.append(4)
    elif(i >= Class_Range * 5 and i < Class_Range * 6): # Heavy metal
        y.append(5)
    elif(i >= Class_Range * 6 and i < Class_Range * 7): # House
        y.append(6)
    elif(i >= Class_Range * 7 and i < Class_Range * 8): # Jazz
        y.append(7)
    elif(i >= Class_Range * 8 and i < Class_Range * 9): # folk
        y.append(8)
    elif(i >= Class_Range * 9 and i < Class_Range * 10): # Country
        y.append(9)

# Divide train set and test set as 60:40
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Feature scaling for optimum performance of machine learning and optimization algorithm
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Principal Component Analysis
pca = PCA(n_components=45)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# SVM model
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train_pca, y_train)

joblib.dump(svm, open(os.path.join(TRAINED_DATA_DIR, 'svm_datas.joblib'),'wb'))
joblib.dump(sc, open(os.path.join(TRAINED_DATA_DIR, 'sc_datas.joblib'),'wb'))
joblib.dump(pca, open(os.path.join(TRAINED_DATA_DIR, 'pca_datas.joblib'),'wb'))

