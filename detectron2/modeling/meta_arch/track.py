import numpy as np
import cv2 as cv
from .detection import *
from .utils import *
import math
class Track(Detection):
    def __init__(self,method,_id,det,frame_gray,conf=0,major_color=[],process_noise=1,measurement_noise=1):
        
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
        
        
        self.missed_count=0
        self.matched=True
        self.descriptor = np.array(np.zeros(det.descriptor.shape[0],np.float32))
        self.descriptor[:] = det.descriptor[:]
        self.track_id = _id
        self.method = method
        self.feature_params=dict(maxCorners=30,qualityLevel=0.3,minDistance=7,blockSize=7)
        self.lk_params=dict(winSize=(15,15),maxLevel=2,criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT,10,0.03))
        if(method=='kalman_vel'):
            self.init_kalman_tracker_vel()
        elif(method=='kalman_acc'):
            self.init_kalman_tracker_acc(measurement_noise,process_noise)
        
            
   
    def init_kalman_tracker_vel(self):
        self.kalman_tracker = cv.KalmanFilter(8,4)
        self.kalman_tracker.measurementMatrix = np.array([[1,0,0,0,0,0,0,0],
                                                          [0,1,0,0,0,0,0,0],
                                                          [0,0,1,0,0,0,0,0],
                                                          [0,0,0,1,0,0,0,0]],np.float32)

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
                                                        [0,0,0,0,0,0,0,1]],np.float32)*0.002

        self.kalman_tracker.predict();

        self.kalman_tracker.correct(self.corners())
        
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
        
    
    def copy(self):
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        
        other = Track(self.track_id,np.array([0,self.conf,xmin,ymin,xmax,ymax],np.float32),self.frame_gray)
        other.tracked_count = self.tracked_count
        other.missed_count = self.missed_count
        other.matched = self.matched
        other.centers = self.centers[:]
        other.areas = self.areas[:]
        other.prev_points = self.prev_points[:]
        other.new_points = self.new_points[:]
        return other
    
    def update(self,det,frame_gray,prev_frame_gray):
        self.matched = True
        self.missed_count = 0
        self.tracked_count +=1
        
        self.old_center = self.center()
        if(len(det.major_color)>0 and len(self.major_color)>0):
            self.major_color[0] = (det.major_color[0]+4*self.major_color[0])/5
        if(self.is_overlap==False):
            self.hog= det.hog
            self.descriptor = 0.7 * self.descriptor + 0.3 *det.descriptor

           
        if(self.method=='kalman_acc' or self.method=='kalman_vel' ):
            self.predict(prev_frame_gray,frame_gray)
            pred = self.kalman_tracker.predict()
            self.kalman_tracker.correct(det.corners())
         
            if(self.tracked_count>15):
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
        #desc_dist = np.linalg.norm(self.hog-get_hog_descriptor(frame_gray,self.pred_xmin,self.pred_ymin,self.pred_xmax,self.pred_ymax),ord=1)
        #if(desc_dist<0.15):
        
        self.old_center = self.center()
        
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
        vec = self.center() - self.old_center
        if(vec[0]==0):
            self.old_slope = 999
        else:
            self.old_slope = vec[1]/vec[0]
        

    def draw_own_mask(self,mask):
        cv.rectangle(mask, (int(self.xmin), int(self.ymin)), (int(self.xmax), int(self.ymax)), (255, 255, 255), -1)
    
    def shiftKeyPointsFlow(self,frame,prev_frame):

        frame_grey = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        frame_width = frame_grey.shape[1]
        frame_height = frame_grey.shape[0]
        prev_frame_grey = cv.cvtColor(prev_frame,cv.COLOR_BGR2GRAY)
        mask = np.zeros(frame_grey.shape, dtype = "uint8")
        cv.rectangle(mask, (int(self.xmin), int(self.ymin)), (int(self.xmax), int(self.ymax)), (255, 255, 255), -1)
        p0 = cv.goodFeaturesToTrack(prev_frame_grey, mask = mask, **self.feature_params)
            
        if(not p0 is None ):
            p1, st, err = cv.calcOpticalFlowPyrLK(prev_frame_grey, frame_grey,p0, None, **self.lk_params)
            old_box = bounding_box_naive(p0)
            new_box = bounding_box_naive(p1)
            
            
            self.new_box = new_box
            offset = [new_box[0]-old_box[0],new_box[1]-old_box[1]]
            
            new_center = self.center() +offset
            old_width = self.xmax - self.xmin
            old_height = self.ymax-self.ymin
            if(old_box[2] ==0):
                old_box[2] = new_box[2]
            if(old_box[3] ==0):
                old_box[3] = new_box[3]
            new_width = old_width * (new_box[2]/old_box[2])
    
            new_height = old_height * (new_box[3]/old_box[3])
            scale_change= (old_width/new_width)*(old_height/new_height)
            if(new_width==0 or new_width>frame_width or math.isnan(new_width)):
                new_width=0
            if(new_height==0 or new_height>frame_height or math.isnan(new_height)):
                new_height=0
            self.offset_tracker.correct(np.array([offset[0],scale_change],np.float32))
            self.offset[0] = offset[0]
            self.offset[1] = scale_change
            self.pred_xmin = new_center[0] - (new_width/2)
            self.pred_ymin=new_center[1] - (new_height/2)
            self.pred_xmax=new_center[0] + (new_width/2)
            self.pred_ymax=new_center[1]+ (new_height/2)
            
        elif(self.missed_count>6):
            self.conf=0
            
        else:
            print('2.2 no points to track')
   
    def predict(self,prev_frame_gray,frame_gray):
        if(self.method=='kalman_vel' or self.method=='kalman_acc'):
            pred = self.kalman_tracker.predict()
            self.pred_xmin = pred[0][0]
            self.pred_ymin = pred[1][0]
            self.pred_xmax = pred[2][0]
            self.pred_ymax = pred[3][0]
            
        elif(self.method =='keypoint_flow'):
            self.shiftKeyPointsFlow(frame_gray,prev_frame_gray)
        elif(self.method =='dense_flow'):
            self.shiftFBFlow()
    def __repr__(self):
        return "id:%d, xmin: %f, ymin:%f, xmax:%f, ymax:%f, conf:%f, class:%d"%(self.track_id, self.xmin,self.ymin,self.xmax,self.ymax,self.conf,self.pred_class)
        
           
            
    