from flask import Flask, render_template, request, send_from_directory 
from werkzeug import secure_filename
import numpy as np
import os
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
    indexing = list(range(0,32))
    name = DataFrame(data, columns =['freq','time'], index = indexing)
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


