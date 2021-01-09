import numpy as np
import cv2 as cv
from .detection import *
from .utils import *
import math
class Track(Detection):
    def __init__(self,method,_id,det,frame_gray,use_kalman=True,conf=0,major_color=[],process_noise=1,measurement_noise=1,embed_alpha = 0.5):
        
        self.conf = conf
        self.xmin = det.xmin
        self.ymin = det.ymin
        self.xmax = det.xmax
        self.ymax = det.ymax
        self.pred_class = det.pred_class
        self.pred_xmin = det.xmin
        self.pred_ymin = det.ymin
        self.pred_xmax = det.xmax
        self.pred_ymax = det.ymax
        self.tracked_count = 1
        self.hog = det.hog
        self.frame_gray = frame_gray
        self.old_center = []
        self.old_slope = 0
        self.major_color=major_color
        self.embed_alpha = embed_alpha
        self.occluded = False
        self.missed_count=0
        self.use_kalman = use_kalman
        self.matched=True
        self.descriptor = np.array(np.zeros(det.descriptor.shape[0],np.float32))
        self.descriptor[:] = det.descriptor[:]
        self.track_id = _id
        self.method = method
       
        if(method=='kalman_vel'):
            self.init_kalman_tracker_vel(measurement_noise,process_noise)
        elif(method=='kalman_acc'):
            self.init_kalman_tracker_acc(measurement_noise,process_noise)
        
            
   
    def init_kalman_tracker_vel(self,measurement_noise,process_noise):
        self.kalman_tracker = cv.KalmanFilter(8,4)
        self.kalman_tracker.measurementMatrix = np.array([[1,0,0,0,0,0,0,0],
                                                          [0,1,0,0,0,0,0,0],
                                                          [0,0,1,0,0,0,0,0],
                                                          [0,0,0,1,0,0,0,0]],np.float32)
        self.kalman_tracker.measurementNoiseCov = np.array([[1,0,0,0],
                                                            [0,1,0,0],
                                                            [0,0,1,0],
                                                            [0,0,0,1]],np.float32)*measurement_noise

        self.kalman_tracker.transitionMatrix = np.array([[1,0,0,0,1,0,0,0],
                                                         [0,1,0,0,0,1,0,0],
                                                         [0,0,1,0,0,0,1,0],
                                                         [0,0,0,1,0,0,0,1],
                                                         [0,0,0,0,1,0,0,0],
                                                         [0,0,0,0,0,1,0,0],
                                                         [0,0,0,0,0,0,1,0],
                                                         [0,0,0,0,0,0,0,1]],np.float32)

        self.kalman_tracker.processNoiseCov = np.array([[1,0,0,0,0,0,0,0],
                                                        [0,1,0,0,0,0,0,0],
                                                        [0,0,1,0,0,0,0,0],
                                                        [0,0,0,1,0,0,0,0],
                                                        [0,0,0,0,1,0,0,0],
                                                        [0,0,0,0,0,1,0,0],
                                                        [0,0,0,0,0,0,1,0],
                                                        [0,0,0,0,0,0,0,1]],np.float32)*process_noise

        self.kalman_tracker.predict();
        self.kalman_tracker.correct(self.corners())
        self.kalman_tracker.correct(convert_bbox_to_z(self.corners()))
        
    def init_kalman_tracker_acc(self,measurement_noise,process_noise):
        self.kalman_tracker = cv.KalmanFilter(12,4)
        self.kalman_tracker.measurementMatrix = np.array([[1,0,0,0,0,0,0,0,0,0,0,0],
                                                          [0,1,0,0,0,0,0,0,0,0,0,0],
                                                          [0,0,1,0,0,0,0,0,0,0,0,0],
                                                          [0,0,0,1,0,0,0,0,0,0,0,0]],np.float32)
        self.kalman_tracker.measurementNoiseCov = np.array([[1,0,0,0],
                                                            [0,1,0,0],
                                                            [0,0,1,0],
                                                            [0,0,0,1]],np.float32)*measurement_noise
        self.kalman_tracker.transitionMatrix = np.array([[1,0,0,0,1,0,0,0,0.5,0,0,0],
                                                         [0,1,0,0,0,1,0,0,0,0.5,0,0],
                                                         [0,0,1,0,0,0,1,0,0,0,0.5,0],
                                                         [0,0,0,1,0,0,0,1,0,0,0,0.5],
                                                         [0,0,0,0,1,0,0,0,0,0,0,0],
                                                         [0,0,0,0,0,1,0,0,0,0,0,0],
                                                         [0,0,0,0,0,0,1,0,0,0,0,0],
                                                         [0,0,0,0,0,0,0,1,0,0,0,0],
                                                         [0,0,0,0,0,0,0,0,1,0,0,0],
                                                         [0,0,0,0,0,0,0,0,0,1,0,0],
                                                         [0,0,0,0,0,0,0,0,0,0,1,0],
                                                         [0,0,0,0,0,0,0,0,0,0,0,1]],np.float32)

        self.kalman_tracker.processNoiseCov = np.array([[1,0,0,0,0,0,0,0,0,0,0,0],
                                                        [0,1,0,0,0,0,0,0,0,0,0,0],
                                                        [0,0,1,0,0,0,0,0,0,0,0,0],
                                                        [0,0,0,1,0,0,0,0,0,0,0,0],
                                                        [0,0,0,0,1,0,0,0,0,0,0,0],
                                                        [0,0,0,0,0,1,0,0,0,0,0,0],
                                                        [0,0,0,0,0,0,1,0,0,0,0,0],
                                                        [0,0,0,0,0,0,0,1,0,0,0,0],
                                                        [0,0,0,0,0,0,0,0,1,0,0,0],
                                                        [0,0,0,0,0,0,0,0,0,1,0,0],
                                                        [0,0,0,0,0,0,0,0,0,0,1,0],
                                                        [0,0,0,0,0,0,0,0,0,0,0,1]],np.float32)*process_noise
     
                                                                 
                                                        

        self.kalman_tracker.predict();
        self.kalman_tracker.correct(self.corners())
        #self.kalman_tracker.correct(convert_bbox_to_z(self.corners()))
        
    
    def copy(self):
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        det = Detection(self.conf,np.array([xmin,ymin,xmax,ymax]),self.pred_class,self.descriptor)
        other = Track(self.method,self.track_id,det,self.frame_gray)
        other.tracked_count = self.tracked_count
        other.missed_count = self.missed_count
        
        return other
    
    def update(self,det,frame_gray,prev_frame_gray):
        self.matched = True
        self.missed_count = 0
        self.tracked_count +=1
        
        self.old_center = self.center()
        if(len(det.major_color)>0 and len(self.major_color)>0):
            self.major_color[0] = (det.major_color[0]+4*self.major_color[0])/5
        
        self.hog= det.hog
        self.descriptor = self.embed_alpha * self.descriptor + (1-self.embed_alpha) *det.descriptor

           
        if(self.method=='kalman_acc' or self.method=='kalman_vel' ):
            self.predict(prev_frame_gray,frame_gray)
            pred = self.kalman_tracker.predict()
            #self.kalman_tracker.correct(convert_bbox_to_z(det.corners()))
            self.kalman_tracker.correct(det.corners())
            
            if(self.use_kalman and self.tracked_count>15):
                self.xmin = self.pred_xmin
                self.ymin = self.pred_ymin
                self.xmax=self.pred_xmax
                self.ymax = self.pred_ymax
                
                
            else:
                self.xmin = det.xmin
                self.ymin = det.ymin
                self.xmax=det.xmax
                self.ymax = det.ymax
              
    
    def apply_prediction(self,frame_gray,prev_frame_gray):
        
        self.predict(prev_frame_gray,frame_gray)
        
        
        
        
        
        if(self.tracked_count>5):
            self.xmin = self.pred_xmin
            self.ymin = self.pred_ymin
            self.xmax = self.pred_xmax
            self.ymax = self.pred_ymax
        elif(self.missed_count>0 and self.missed_count<3):
            self.xmin = self.pred_xmin
            self.ymin = self.pred_ymin
            self.xmax = self.pred_xmax
            self.ymax = self.pred_ymax
        
        
            
        
        

    def draw_own_mask(self,mask):
        cv.rectangle(mask, (int(self.xmin), int(self.ymin)), (int(self.xmax), int(self.ymax)), (255, 255, 255), -1)
    
    
    def predict(self,prev_frame_gray,frame_gray):
        if(self.method=='kalman_vel' or self.method=='kalman_acc'):
            pred = self.kalman_tracker.predict()
            
            #pred = pred.reshape(1,8)[0]
            #pred_box = convert_x_to_bbox(pred[0:4])
            
            #print(pred_box)
            self.pred_xmin = pred[0][0]
            self.pred_ymin = pred[1][0]
            self.pred_xmax = pred[2][0]
            self.pred_ymax = pred[3][0]
            
        
    def __repr__(self):
        return "id:%d, xmin: %f, ymin:%f, xmax:%f, ymax:%f, conf:%f, class:%d"%(self.track_id, self.xmin,self.ymin,self.xmax,self.ymax,self.conf,self.pred_class)
        
           
            
    