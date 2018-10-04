from flask import Flask, render_template, request, send_from_directory 
from werkzeug import secure_filename
import librosa
import numpy as np
import os
from pydub import AudioSegment
import pickle
import tempfile
from pandas import Series, DataFrame
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def createTable(name):
    data={}
    name = DataFrame(data, columns =['freq','time'],
                     index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
                     20,21,22,23,24,25,26,27,28,29,30,31])
    return name


def addData(temp_freq,freqN,n):
    temp_freq = temp_freq.sort_values(by=['freq'],axis=0,ascending=False)
    freqN.ix[n] = temp_freq.ix[0]
    return freqN

def sortCandidate(table,ascending):
    if(ascending == False):
        table = table.sort_values(by=['freq'],axis=0,ascending=False)
    elif(ascending == True):
        table = table.sort_values(by=['time'],axis=0,ascending=True)

    return table

def tableIloc(table):
    table = table.iloc[:12]
    
    return table

def tableRename(table):

    table = table.rename(index={table.index[0]:0,table.index[1]:1,table.index[2]:2,table.index[3]:3,
                                table.index[4]:4,table.index[5]:5,table.index[6]:6,table.index[7]:7,
                                table.index[8]:8,table.index[9]:9,table.index[10]:10,table.index[11]:11,})
    
    return table

def deleteCandidate(table,temp):
    num = len(temp)-1
    while(num>-1):        
        table.ix[temp[num]]=[np.NaN,np.NaN]
        num -= 1
        table = tableRename(table)
        
    return table

def choiceCandidate(table):
    for i in range(0, len(table.index)):
        temp = []       
        for j in range(i+1,len(table.index)):                        
            if(table.ix[i].time < table.ix[j].time+32 and table.ix[i].time > table.ix[j].time-32):
                temp.append(j)                               
        table = deleteCandidate(table,temp)

    return table


f = tempfile.mktemp()

svm = pickle.load(open(os.path.join('svm_data', 'svm_datas.pkl'),'rb'))
sc = pickle.load(open(os.path.join('svm_data', 'sc_datas.pkl'),'rb'))
pca = pickle.load(open(os.path.join('svm_data', 'pca_datas.pkl'),'rb'))


app = Flask(__name__)

UPLOAD_FOLDER = 'C:/Users/netpi/python/flask_upload' 
ALLOWED_EXTENSIONS = set(['mp3','wav']) 

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

@app.route('/upload')
def File_upload():
    return render_template('upload.html')

def classify(file_name):
    sound = AudioSegment.from_mp3(UPLOAD_FOLDER + '/%s' % (file_name))
    sound.export(f, format="wav")
    
    data = [[0]*254]

    hop_length = 512

    y, sr = librosa.load(f, offset = 40)
    
    y_harmonic, y_percussive = librosa.effects.hpss(y) 
    
    per_stft= librosa.core.stft(y,n_fft=256)
    
    sample = [[0]*1024 for i in range(128)]
    for i in range(0,128):
        for j in range(0,1024):
            sample[i][j] = per_stft[i][j]
            
    freq37 = createTable('freq37')
    freq38 = createTable('freq38')
    freq39 = createTable('freq39')
    freq40 = createTable('freq40')
    freq41 = createTable('freq41')
    freq42 = createTable('freq42')
    freq43 = createTable('freq43')
    temp_freq = createTable('temp_freq')

    freqTable = []
    freqTable.append(freq37)
    freqTable.append(freq38)
    freqTable.append(freq39)
    freqTable.append(freq40)
    freqTable.append(freq41)
    freqTable.append(freq42)
    freqTable.append(freq43)

    for i in range(0,7):
        n=0
        for j in range(0,1024):        
            temp_freq.ix[j%32] = [sample[i][j],j]
            if(j != 0 and j%32 == 0):
                freqTable[i] = addData(temp_freq,freqTable[i],n)
                n += 1
            elif(j==1023):
                freqTable[i] = addData(temp_freq,freqTable[i],n)
                n += 1
                
    for i in range(7):    
        freqTable[i] = sortCandidate(freqTable[i],False)    
        freqTable[i] = tableIloc(freqTable[i])    
        freqTable[i] = tableRename(freqTable[i])   
        freqTable[i] = choiceCandidate(freqTable[i])
        freqTable[i] = sortCandidate(freqTable[i],True)
        freqTable[i] = tableRename(freqTable[i])
        
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
                time_list.append(np.mean(temp))

    time_list.sort()
    
    time_interval=[]

    for i in range(len(time_list)-1):            
        time_interval.append(time_list[i+1] - time_list[i])
    
    if(len(time_interval)<=1):
        time_interval.append(861)
    
    time_interval.sort()   
    
    if(len(time_interval)>2):
        time_interval.pop(len(time_interval)-1)

    if(len(time_interval)>2):
        time_interval.pop(0)

    bar_length = np.mean(time_interval)*2*256

    #bar_start = (max(time_interval)-(np.mean(time_interval)/2))*256
    bar_start = bar_length/4

    climax_length = bar_length*8
    
    max_wave=max(y)
    music_length = y.shape
    bar_num = int(music_length/bar_length)+1

    music_divide = []
    n=0
    for i in range(0,bar_num):
        music_divide.append(y[n:int(bar_length)*(i+1)])
        n=n+int(bar_length)

    save_count = []

    for i in range(0,bar_num):
        count = 0
        for j in range(0,len(music_divide[i])):
            if(music_divide[i][j] >= max_wave*0.9):
                count += 1
        save_count.append(count)        

    bar_climax= save_count.index(max(save_count))
    climax = bar_climax*bar_length
    
    music_start = climax/sr
    music_duration = (climax/sr + climax_length/sr) - climax/sr
    realbar_start = bar_start/sr
    
    y, sr = librosa.load(f, offset = music_start+40-realbar_start, duration = music_duration)
    
    
    #MFCC
    
    p_mfcc = librosa.feature.mfcc(y=y_percussive, sr=sr, hop_length = hop_length, n_mfcc=20)
    h_mfcc = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=20)
    
    for j in range(0,20):
        data[0][j] = np.mean(p_mfcc[j])    
        data[0][j+20] = np.var(p_mfcc[j]) 
        data[0][j+40] = np.mean(h_mfcc[j])
        data[0][j+60] = np.var(h_mfcc[j])
               
    #dMFCC
    d_p_mfcc = librosa.feature.delta(p_mfcc)
    d_h_mfcc = librosa.feature.delta(h_mfcc)
    
    #ddMFCC
    d2_p_mfcc = librosa.feature.delta(p_mfcc, order=2)
    d2_h_mfcc = librosa.feature.delta(h_mfcc, order=2)
    
    for j in range(0,20):
        data[0][j+80] = np.mean(d_p_mfcc[j])
        data[0][j+100] = np.var(d_p_mfcc[j])
        data[0][j+120] = np.mean(d_h_mfcc[j])
        data[0][j+140] = np.var(d_h_mfcc[j])
        data[0][j+160] = np.mean(d2_p_mfcc[j])
        data[0][j+180] = np.var(d2_p_mfcc[j])
        data[0][j+200] = np.mean(d2_h_mfcc[j])
        data[0][j+220] = np.var(d2_h_mfcc[j])
        
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
    

def allowed_file(filename): 
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS
    
@app.route('/', methods=['GET','POST']) 
def upload_file(): 
    if request.method == 'POST': 
        file = request.files['file'] 
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename) 
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename)) 
            result = classify(filename);
            return '%s 음악에 대한 분류 : %s' % (filename ,result)
    return render_template("upload.html") 

if __name__ == '__main__':
    app.run()


