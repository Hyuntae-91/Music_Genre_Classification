#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from modules import Climax

def Classify():
    sound = AudioSegment.from_mp3(UPLOAD_FOLDER + '/%s' % (file_name))
    sound.export(f, format="wav")

    data = [[0]*254]

    hop_length = 512

    # y is a variable that expresses audio as time
    # sr is sampling rate
    y, sr = librosa.load(f, offset = 40) 

    # Decompose an audio time series into harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y) 

    # stft is short time fourier transform function
    per_stft= librosa.core.stft(y,n_fft=256) 

    # Because the waveform is symmetric, only 128 waveform are extracted
    # Insert 128 * 1024 samples at the list
    sample = [[0]*1024 for i in range(128)]  # extract snare drum
    for i in range(0,128):
        for j in range(0,1024):
            sample[i][j] = per_stft[i][j]

    # Load climax module
    music_start, music_duration, realbar_start = Climax(sample)


    # Reload music from climax
    y, sr = librosa.load(f, offset = music_start+40-realbar_start, duration = music_duration)

    #MFCC
    p_mfcc = librosa.feature.mfcc(y=y_percussive, sr=sr, hop_length = hop_length, n_mfcc=20)
    h_mfcc = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=20)

    for i in range(0,20):
        data[0][i] = np.mean(p_mfcc[i])
        data[0][i+20] = np.var(p_mfcc[i])
        data[0][i+40] = np.mean(h_mfcc[i])
        data[0][i+60] = np.var(h_mfcc[i])

    #dMFCC
    d_p_mfcc = librosa.feature.delta(p_mfcc)
    d_h_mfcc = librosa.feature.delta(h_mfcc)

    #ddMFCC
    d2_p_mfcc = librosa.feature.delta(p_mfcc, order=2)
    d2_h_mfcc = librosa.feature.delta(h_mfcc, order=2)

    for i in range(0,20):
        data[0][i+80] = np.mean(d_p_mfcc[i])
        data[0][i+100] = np.var(d_p_mfcc[i])
        data[0][i+120] = np.mean(d_h_mfcc[i])
        data[0][i+140] = np.var(d_h_mfcc[i])
        data[0][i+160] = np.mean(d2_p_mfcc[i])
        data[0][i+180] = np.var(d2_p_mfcc[i])
        data[0][i+200] = np.mean(d2_h_mfcc[i])
        data[0][i+220] = np.var(d2_h_mfcc[i])

    #tempo, beat_frames
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

    #beat_times
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    #beat_mfcc_delta
    beat_mfcc_delta = librosa.util.sync(np.vstack([p_mfcc, d_p_mfcc]), beat_frames)

    chromagram = librosa.feature.chroma_cqt(y=y_harmonic,sr=sr)

    # Aggregate chroma features between beat events
    beat_chroma = librosa.util.sync(chromagram, beat_frames,aggregate=np.median)

    # Finally, stack all beat-synchronous features together
    beat_features = np.vstack([beat_chroma, beat_mfcc_delta])
    data[0][240] = np.mean(tempo)
    data[0][241] = np.var(tempo)
    data[0][242] = np.mean(beat_times)
    data[0][243] = np.var(beat_times)
    data[0][244] = np.mean(beat_frames)
    data[0][245] = np.var(beat_frames)
    data[0][246] = np.mean(beat_mfcc_delta)
    data[0][247] = np.var(beat_mfcc_delta)
    data[0][248] = np.mean(chromagram)
    data[0][249] = np.var(chromagram)
    data[0][250] = np.mean(beat_chroma)
    data[0][251] = np.var(beat_chroma)
    data[0][252] = np.mean(beat_features)
    data[0][253] = np.var(beat_features)

    X_test_std = sc.transform(data)
    X_test_pca = pca.transform(X_test_std)

    y_pred = svm.predict(X_test_pca)
    return y_pred
