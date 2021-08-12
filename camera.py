import cv2
from model import FacialExpressionModel
import numpy as np
from threading import Thread
from datetime import datetime
from time import sleep

class DataRetriever:
    def __init__(self,src=0):
        self.rgb =  cv2.VideoCapture(src)
        self.fps = self.rgb.get(cv2.cv2.CAP_PROP_FPS)
        self.count = 0

    def start(self):
        Thread(target=self.__get_data__, args=()).start()
        return self

    def __get_data__(self):
        (_, self.frame) = self.rgb.read()
        self.count+=1
    
    def close(self):
        self.rgb.release()

class Displayer:
    def __init__(self,frame=np.zeros((480,640))):
        self.frame = frame
        self.done = False
        self.count = 0
    
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


class Predicter:
    def __init__(self,cnn,displayer):
        self.frame = None
        self.faces = None
        self.gray = None
        self.newFrame = False
        self.done = False
        self.count = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.cnn = cnn
        self.displayer = displayer
        self.facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        #list where the images used to write the video files are stored
        self.predictions = []


    
    def start(self,):
        Thread(target=self.predict,args=()).start()
        return self

    def predict(self):
        while not self.done:
            #when a new frame is grabed a prediction is made
            if self.newFrame:
                self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                self.faces = self.facec.detectMultiScale(self.gray, 1.3, 5)
                for (x, y, w, h) in self.faces:
                        fc = self.gray[y:y+h, x:x+w]
                        
                        roi = cv2.resize(fc, (48, 48))
                        #roi = cv2.normalize(roi,None)
                        pred = self.cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

                        cv2.putText(self.frame, pred, (x, y), self.font, 1, (255, 255, 0), 2)
                        cv2.rectangle(self.frame,(x,y),(x+w,y+h),(255,0,0),2)
                self.displayer.frame = self.frame
                #cv2.imwrite('f'+str(self.count).zfill(4) + '.jpg',self.frame)
                self.count +=1
                self.predictions.append(self.frame)
                self.newFrame = False

    def set_frame(self,frame):
        self.frame = frame
        self.newFrame = True        
    
    def stop(self,):
        self.done = True
        #at the end of the program, when there are no more frames to check, write all the predictions
        #into a video file 
        out = cv2.VideoWriter('FER.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640,480))
 
        for i in range(len(self.predictions)):
            vidout=cv2.resize(self.predictions[i],(640,480)) 
            out.write(vidout)
        out.release()


#app runs on three threads: one to grab the data, one to find faces and predict emotions and finally one to display results
def start_app(cnn,src=0):
    startTime = datetime.now()
    #the variable runningTime specifies the amount of time the camera should run 
    #the variable is set to 60 seconds taking into consideration the setup time requirement 
    #and the 30 seconds actually used to grab images
    runningTime = 60

    #start threads
    data_retriver = DataRetriever(src).start()
    displayer = Displayer().start()
    predicter = Predicter(cnn,displayer).start()

    #actual program
    while (datetime.now()-startTime).seconds<runningTime:
        data_retriver.__get_data__()
        predicter.set_frame(data_retriver.frame)

    #end threads
    data_retriver.close()
    predicter.stop()
    displayer.stop()

if __name__ == '__main__':
    model = FacialExpressionModel("face_model.json", "face_model.h5")
    start_app(model)