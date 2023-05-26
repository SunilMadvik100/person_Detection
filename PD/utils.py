import os
import cv2
import shutil
import numpy as np
import logging
import pandas as pd
from time import time
from datetime import datetime
from Configurations import Configurations


def GetTodayDate():
    return datetime.now().strftime('%m/%d/%Y')

def GetTodayDateLogs():
    return datetime.now().strftime('%d-%m-%Y')

def GetTimeNow():
    return datetime.now().strftime('%H:%M:%S')

def SaveImg(path, img):
    cv2.imwrite(path, img)


def RemoveFolder(path):
    shutil.rmtree(path)

def CreateFolder(path):
    if os.path.exists(path):
        RemoveFolder(path)
    os.mkdir(path)

def CreateFolderForLogs(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

class CalcTime(object):

    def start(self):
        self.begin = time()

    def end(self):
        self.finish = time()

    def calculate(self):
        return np.round(self.finish-self.begin, 2)

    def reset(self):
        self.begin = 0
        self.finish = 0


class Logs(object):

    def __init__(self):
        self.logs = logging
        self.time_recorder = CalcTime()
        self.CreateLogs()
        return

    def setlogclass(self, classname):
        self.logs.getLogger(classname)

    def CreateLogs(self):
        CreateFolderForLogs('Logs')
        self.logs.basicConfig(filename=os.path.join('Logs', GetTodayDateLogs() + '.log'), format= "[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s",level=logging.INFO)

    def UpdateLogs(self, message):
        self.logs.info(message)

    def UpdateErrorLogs(self, message):
        self.logs.error(message)



