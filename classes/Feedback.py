from PyQt5 import QtWidgets, uic
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QDesktopWidget, QLineEdit, QPushButton, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import time
import random
import shutil
from random import seed
from random import randint


seed(1)
classes = ('Angry','Happy','Sad','Surprise','Neutral')
FILENAME = 'Analysed_Data.csv'
QLine = 0

#Erstellt Klassifikation. Wählt aus Angry, Happy, Sad, Surprise, Neutral
def createRndAudio():
    emotion = random.choice(classes)
    return emotion

def createRndEmo():
    emotion = randint(0,4)
    return emotion

#Erstellt random Engagement Score
def createRndVisual():
    e_score = randint(0,100)
    return e_score

#Erstellt csv File mit der ersten Zeile welche die Spalten definiert (Time Stamp,Engagement Score, Angry, Happy, Sad, Surprise, Neutral)
def createCsvFile():
    with open(FILENAME, 'w') as file:
        file.write(
            'Time Stamp,Engagement Score,Angry,Happy,Sad,Surprise,Neutral\n')


#Ergänzt CSV File  mit Time Stamp, Attention Score und Emotion
def appendCsvFile(time, e_score, emotion=-1):

    text = []
    corrLine = []

    #read file lines into text
    with open(FILENAME, 'r') as file:
        for line in file:
            text.append(line)

    #check if new lines time stamp already exists in file
    foundLine = False
    i = 0
    for line in text:
        compTime = text[i].find(str(time), 0, 25)
        if compTime != -1:
            foundLine = True
            corrLine = text[i].rstrip('\n')
            break
        i += 1

    #if new line already exists add emotion value
    if foundLine == True:
        if emotion == 0:
            lineEnd = corrLine[-8:]
            temp = corrLine[-9:-8]
            inttemp = int(temp)
            inttemp += 1
            corrLine = corrLine[:-9] + str(inttemp) + lineEnd + '\n'
            
        elif emotion == 1:
            lineEnd = corrLine[-6:]
            temp = corrLine[-7:-6]
            inttemp = int(temp)
            inttemp += 1
            corrLine = corrLine[:-7] + str(inttemp) + lineEnd + '\n'
            
        elif emotion == 2:
            lineEnd = corrLine[-4:]
            temp = corrLine[-5:-4]
            inttemp = int(temp)
            inttemp += 1
            corrLine = corrLine[:-5] + str(inttemp) + lineEnd + '\n'        

        elif emotion == 3:
            lineEnd = corrLine[-2:]
            temp = corrLine[-3:-2]
            inttemp = int(temp)
            inttemp += 1
            corrLine = corrLine[:-3] + str(inttemp) + lineEnd + '\n'
            
        elif emotion == 4:
            temp = corrLine[-1:]
            inttemp = int(temp)
            inttemp += 1
            corrLine = corrLine[:-1] + str(inttemp)+'\n'          

        text[i] = corrLine
        
        with open(FILENAME, 'w') as file:
            for line in text:
                file.write(line)
        
    #if new line doesnt exist already append new line into file
    if foundLine == False:
        with open(FILENAME, 'a') as file:
            file.write(str(time))       # Time Stamp
            file.write(',')
            file.write(str(e_score))    # Attention score

            for k in range(5):          # Emotion
                file.write(',')
                if (emotion == k):
                    file.write(str(1))
                else:
                    file.write(str(0))
                    
            file.write('\n')

def textchanged(text):
    global QLine
    QLine = int(text)


def getCsvLine(time):

    text = []
    corrLine = []

    #read file lines into text
    with open(FILENAME, 'r') as file:
        for line in file:
            text.append(line)
    
    
    for line in text:
        corrTime = line.find(str(time), 0, len(str(time)))
        if corrTime != -1:
            corrLine = line.rstrip('\n')
            break

    if corrTime == -1:
        print("No matching line found in Csv File")
        corrLine = ("0.0,0.0,0,0,0,0,0")

    return corrLine


class FeedbackWindow(QMainWindow):
    def __init__(self, DataName):
        super(FeedbackWindow, self).__init__()
        self.df = pd.read_csv(DataName, engine="python")
        self.layout = QGridLayout()
        self.setWindowTitle("Feedback")
        self.setStyleSheet("background-color: black;")

        self.initUI()

    def initUI(self):

        self.setGeometry(0, 0, 1920, 1080)
        self.center()

        self.plotAttention()

        self.plotEmotionSum()

        self.EmotionFrameLineWidget()

        self.plotEmotionFrame("0.0,0.0,0,0,0,0,0")

        self.l1 = QLabel()
        self.l1.setText("The most common Emotion was:")
        self.l1.setStyleSheet("background-color: grey")
        self.l1.setFont(QFont('Arial', 15))
        self.l1.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.l1, 2, 1)

        self.saveButton = QPushButton('Save Meeting Analysis', self)
        self.saveButton.setToolTip('Save your meeting Analysis to compare with future Meetings!')
        self.saveButton.setStyleSheet("background-color: lightgrey")
        self.saveButton.clicked.connect(self.clickSafeButton)
        self.layout.addWidget(self.saveButton, 4,0)

        self.addResult()

        '''
        self.plotAngry()
        self.plotHappy()
        self.plotSad()
        self.plotSurprise()
        self.plotNeutral()
        '''

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
           

    def center(self):

        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def clickSafeButton(self):
        timestr = time.strftime("Meeting_%Y.%m.%d-%H%M")
        self.saveButton.setEnabled(False)
        if not os.path.exists('Saves'):
            os.makedirs('Saves')
        dstFile = os.path.abspath('Saves/'+timestr+'.csv')
        srcFile = os.path.abspath(FILENAME)
        shutil.copy(srcFile, dstFile)

    def plotAttention(self):

        timeStampList = self.df["Time Stamp"].tolist()
        attentionList = self.df["Engagement Score"].tolist()
        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setYRange(0,1)
        self.graphWidget.setTitle("Attention")
        self.graphWidget.setLabel('left', 'Attention Score')
        self.graphWidget.setLabel('bottom', 'Time')
        self.graphWidget.showGrid(x=True, y=True)
        self.graphWidget.plot(timeStampList, attentionList)
        self.layout.addWidget(self.graphWidget, 0, 0)


    def plotEmotionSum(self):

        angryCount = sum(self.df["Angry"].tolist())
        happyCount = sum(self.df["Happy"].tolist())
        sadCount = sum(self.df["Sad"].tolist())
        surpriseCount = sum(self.df["Surprise"].tolist())
        neutralCount = sum(self.df["Neutral"].tolist())

        emotionGraph = pg.BarGraphItem(x=range(5), height=[angryCount, happyCount, sadCount, surpriseCount, neutralCount], width=0.5)

        ticks = [list(zip(range(5), ('Angry', 'Happy', 'Sad', 'Surprise', 'Neutral')))]


        self.graphWidget = pg.PlotWidget()
        xax = self.graphWidget.getAxis('bottom')
        xax.setTicks(ticks)
        self.graphWidget.addItem(emotionGraph)
        self.graphWidget.setXRange(-0.5, 4.5)
        self.graphWidget.setTitle("Emotions summed")
        self.graphWidget.setLabel('left', 'Number of Emotions')
        self.graphWidget.setLabel('bottom', 'Emotion')
        self.graphWidget.showGrid(x=True, y=True)
        self.layout.addWidget(self.graphWidget, 0, 1)

    def addResult(self):

        angryCount = sum(self.df["Angry"].tolist())
        happyCount = sum(self.df["Happy"].tolist())
        sadCount = sum(self.df["Sad"].tolist())
        surpriseCount = sum(self.df["Surprise"].tolist())
        neutralCount = sum(self.df["Neutral"].tolist())

        resultDic = {'Angry': angryCount, 'Happy': happyCount, 'Sad': sadCount, 'Surprise': surpriseCount, 'neutral': neutralCount}
        result = max(resultDic, key = resultDic.get)

        self.l = QLabel()
        self.l.setText(str(result))
        self.l.setStyleSheet("color : lightgrey")
        self.l.setFont(QFont('Calibri', 30))
        self.l.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.l, 3, 1)

    def EmotionFrameLineWidget(self):
        self.e = QLineEdit()
        self.e.setMaxLength(10)
        self.e.setPlaceholderText("Enter Time Stamp")
        self.e.setStyleSheet("background-color: white;")
        self.layout.addWidget(self.e, 1, 2)

        self.e.textChanged.connect(textchanged)
        self.e.editingFinished.connect(self.enterPress)

    def enterPress(self):

        time = getCsvLine(QLine)
        self.plotEmotionFrame(time)


    def plotEmotionFrame(self, frame):

        angry = int(frame[-9:-8])
        happy = int(frame[-7:-6])
        sad = int(frame[-5:-4])
        surprise = int(frame[-3:-2])
        neutral = int(frame[-1:])

        emotionGraph = pg.BarGraphItem(x=range(5), height=[angry, happy, sad, surprise, neutral], width=0.5)

        ticks = [list(zip(range(5), ('Angry', 'Happy', 'Sad', 'Surprise', 'Neutral')))]

        self.graphWidget = pg.PlotWidget()
        xax = self.graphWidget.getAxis('bottom')
        xax.setTicks(ticks)
        self.graphWidget.addItem(emotionGraph)
        self.graphWidget.setXRange(-0.5, 4.5)
        self.graphWidget.setYRange(0, 1)
        self.graphWidget.setTitle("Emotion per frame")
        #self.graphWidget.setLabel('left', '')
        self.graphWidget.setLabel('bottom', 'Emotion')
        self.layout.addWidget(self.graphWidget, 0, 2)


    def plotAngry(self):

        timeStampList = self.df["Time Stamp"].tolist()
        #timeStampList = timeStampList[::5]
        angryList = self.df["Angry"].tolist()
        #angryList = angryList[::5]
        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setTitle("Angry")
        self.graphWidget.setLabel('left', 'Angry Faces recognized')
        self.graphWidget.setLabel('bottom', 'Time')
        self.graphWidget.plot(timeStampList, angryList)
        self.layout.addWidget(self.graphWidget, 2, 0)


    def plotHappy(self):

        timeStampList = self.df["Time Stamp"].tolist()
        happyList = self.df["Happy"].tolist()
        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setTitle("Happy")
        self.graphWidget.setLabel('left', 'Happy Faces recognized')
        self.graphWidget.setLabel('bottom', 'Time')
        self.graphWidget.plot(timeStampList, happyList)
        self.layout.addWidget(self.graphWidget, 2, 1)

    
    def plotSad(self):

        timeStampList = self.df["Time Stamp"].tolist()
        sadList = self.df["Sad"].tolist()
        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setTitle("Sad")
        self.graphWidget.setLabel('left', 'Sad Faces recognized')
        self.graphWidget.setLabel('bottom', 'Time')
        self.graphWidget.plot(timeStampList, sadList)
        self.layout.addWidget(self.graphWidget, 2, 2)


    def plotSurprise(self):

        timeStampList = self.df["Time Stamp"].tolist()
        surpriseList = self.df["Surprise"].tolist()
        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setTitle("Surprise")
        self.graphWidget.setLabel('left', 'Surprise Faces recognized')
        self.graphWidget.setLabel('bottom', 'Time')
        self.graphWidget.plot(timeStampList, surpriseList)
        self.layout.addWidget(self.graphWidget, 3, 0)


    def plotNeutral(self):

        timeStampList = self.df["Time Stamp"].tolist()
        neutralList = self.df["Neutral"].tolist()
        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setTitle("Neutral")
        self.graphWidget.setLabel('left', 'Neutral Faces recognized')
        self.graphWidget.setLabel('bottom', 'Time')
        self.graphWidget.plot(timeStampList, neutralList)
        self.layout.addWidget(self.graphWidget, 3, 1)
        

