#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import joblib
import os
from .Extract_Feature import Extract_Feature
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

TRAINED_DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) + "/Trained_Data_Set"

def Classifier(Music_data):
    Features = [[0]*254]

    Features = Extract_Feature(Music_data)

    # Load saved model
    svm = joblib.load(open(os.path.join(TRAINED_DATA_DIR, 'svm_data.joblib'), 'rb'))
    sc = joblib.load(open(os.path.join(TRAINED_DATA_DIR, 'sc_data.joblib'), 'rb'))
    pca = joblib.load(open(os.path.join(TRAINED_DATA_DIR, 'pca_data.joblib'), 'rb'))

    # Feature scaling and Principal Component Analysis
    X_test_std = sc.transform(Features)
    X_test_pca = pca.transform(X_test_std)

    # Predict the music genre based with loaded model
    y_pred = svm.predict(X_test_pca)
    return y_pred
