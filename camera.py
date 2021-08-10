import cv2
from model import FacialExpressionModel
import numpy as np
from threading import Thread
from datetime import datetime

# rgb = cv2.VideoCapture(0)
# facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# font = cv2.FONT_HERSHEY_SIMPLEX

class DataRetriever:
    def __init__(self,src=0):
        self.startTime = datetime.now()
        self.runningTime = 30               #seconds
        self.rgb =  cv2.VideoCapture(src)
        self.facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        (_, self.frame) = self.rgb.read()
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.faces = self.facec.detectMultiScale(self.gray, 1.3, 5)

    def start(self):
        Thread(target=self.__get_data__, args=()).start()
        return self

    def __get_data__(self):
        (_, self.frame) = self.rgb.read()
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.faces = self.facec.detectMultiScale(self.gray, 1.3, 5)

class Displayer:
    def __init__(self,frame=np.zeros((48,48))):
        self.frame = frame
        self.done = False
    
    def start(self):
        Thread(target=self.display,args=()).start()
        return self
    def display(self):
        while not self.done:
            cv2.imshow('Filter', self.frame)
            if cv2.waitKey(1) == 27:
                self.done = True
    
    def stop(self):
        self.done = True
        cv2.destroyAllWindows()


def start_app(cnn,src=0):
    data_retriver = DataRetriever(src).start()
    displayer = Displayer().start()
    while (datetime.now()-data_retriver.startTime).seconds<data_retriver.runningTime:
        data_retriver.__get_data__()
        frame = data_retriver.frame
        gray = data_retriver.gray
        faces = data_retriver.faces

        for (x, y, w, h) in faces:
            fc = gray[y:y+h, x:x+w]
            
            roi = cv2.resize(fc, (48, 48))
            pred = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(frame, pred, (x, y), data_retriver.font, 1, (255, 255, 0), 2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        displayer.frame=frame
    displayer.stop()


if __name__ == '__main__':
    model = FacialExpressionModel("face_model.json", "face_model.h5")
    start_app(model)