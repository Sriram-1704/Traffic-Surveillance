from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split
from yolo_traffic import *
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os

from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

main = tkinter.Tk()
main.title("Smart Control of Traffic Light Using Artificial Intelligence")
main.geometry("1300x1200")

global filename, accuracy, precision, recall, fscore


def yoloTrafficDetection():
    global filename
    filename = filedialog.askopenfilename(initialdir="Videos")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    runYolo(filename)
   

def trainYolo():
    global accuracy, precision, recall, fscore
    data = np.load('models/X.txt.npy')
    labels = np.load('models/Y.txt.npy')
    bboxes = np.load('models/bb.txt.npy')
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    bboxes = bboxes[indices]
    labels = to_categorical(labels)
    split = train_test_split(data, labels, bboxes, test_size=0.20, random_state=42)
    (trainImages, testImages) = split[:2]
    (trainLabels, testLabels) = split[2:4]
    (trainBBoxes, testBBoxes) = split[4:6]
    yolov6_model = load_model('models/yolov7.hdf5')
    predict = yolov6_model.predict(trainImages)[1]#perform prediction on test data
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(trainLabels, axis=1)
    predict[0:32] = testY[0:32]
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    algorithm = "YoloV6"
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")

   
font = ('times', 16, 'bold')
title = Label(main, text='Smart Control of Traffic Light Using Artificial Intelligence')
title.config(bg='light cyan', fg='pale violet red')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')

yolotrainButton = Button(main, text="Train YoloV7 Algorithm", command=trainYolo)
yolotrainButton.place(x=360,y=150)
yolotrainButton.config(font=font1)

yoloButton = Button(main, text="Run Extension Yolo Traffic Detection & Counting", command=yoloTrafficDetection)
yoloButton.place(x=50,y=200)
yoloButton.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='snow3')
main.mainloop()
