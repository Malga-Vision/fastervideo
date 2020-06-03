import numpy as np
import cv2 as cv
from .detection import *
from .utils import *
import math
class Track(Detection):
    def __init__(self,method,_id,det,frame_gray,conf=0):
        
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
        self.color=det.color
        self.centers= []
        self.areas = []
        self.prev_points = []
        self.new_points = []
        self.offset = np.array([0,0],np.float32)
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
            self.init_kalman_tracker_acc()
        self.init_offset_tracker()
            
    def init_offset_tracker(self):
        self.offset_tracker= cv.KalmanFilter(4,2)
        self.offset_tracker.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]],np.float32)

        self.offset_tracker.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]],np.float32)

        self.offset_tracker.processNoiseCov = np.array([[1,0,0,0],
                                   [0,1,0,0],
                                   [0,0,1,0],
                                   [0,0,0,1]],np.float32) * 0.001
        self.offset_tracker.correct(self.offset)
        self.offset_tracker.predict();
    
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
                                                        [0,0,0,0,0,0,0,1]],np.float32)

        self.kalman_tracker.predict();

        self.kalman_tracker.correct(self.corners())
        
    def init_kalman_tracker_acc(self):
        self.kalman_tracker = cv.KalmanFilter(12,4)
        self.kalman_tracker.measurementMatrix = np.array([[1,0,0,0,0,0,0,0,0,0,0,0],
                                                          [0,1,0,0,0,0,0,0,0,0,0,0],
                                                          [0,0,1,0,0,0,0,0,0,0,0,0],
                                                          [0,0,0,1,0,0,0,0,0,0,0,0]],np.float32)

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
                                                        [0,0,0,0,0,0,0,0,0,0,0,1]],np.float32)*0.1
        
        self.kalman_tracker.measurementNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)*0.00001

        self.kalman_tracker.predict();

        self.kalman_tracker.correct(self.corners())
        
    
    def copy(self):
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        
        other = Track(self.track_id,np.array([0,self.conf,xmin,ymin,xmax,ymax],np.float32),None)
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
        if(self.is_overlap==False):
            self.descriptor = det.descriptor
            self.hog= det.hog
            self.color=det.color
        if(self.tracked_count>3):
            self.conf = np.maximum(det.conf,0.5)
        else:
            self.conf = 0
       
        if(self.method=='keypoint_flow' or self.method=='dense_flow'):
            
            self.xmin = det.xmin
            self.ymin = det.ymin
            self.xmax=det.xmax
            self.ymax = det.ymax
           
        if(self.method=='kalman_acc' or self.method=='kalman_vel' ):
            self.predict(prev_frame_gray,frame_gray)
            pred = self.kalman_tracker.predict()
            self.kalman_tracker.correct(det.corners())
#           
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
            
                
    def search_local_best_match(self,frame):
        s=8
        shift_x = 0
        shift_y =0
        shifts_x = []
        shifts_y = []
        sel_dist = np.linalg.norm(get_hog_descriptor(frame,self.pred_xmin,self.pred_ymin,self.pred_xmax,self.pred_ymax)-self.hog,ord=1)
        init_dist = sel_dist
        while s>=1:
            shift_x = 0
            shift_y = 0
            for x in (-s,0,s):
                for y in(-s,0,s):
                    
                    dist = np.linalg.norm(get_hog_descriptor(frame,self.pred_xmin+x,self.pred_ymin+y,self.pred_xmax+x,self.pred_ymax+y)-self.hog,ord=1)
                    if(dist<sel_dist):
                       sel_dist = dist
                       shift_x = x
                       shift_y = y
            shifts_x.append(shift_x)
            shifts_y.append(shift_y)
            self.pred_xmin  += shift_x
            self.pred_xmax  += shift_x
            self.pred_ymin  += shift_y
            self.pred_ymax  += shift_y
            s = s/2
        
        
        
    def apply_prediction(self,frame_gray,prev_frame_gray):
        
        self.predict(prev_frame_gray,frame_gray)
        #desc_dist = np.linalg.norm(self.hog-get_hog_descriptor(frame_gray,self.pred_xmin,self.pred_ymin,self.pred_xmax,self.pred_ymax),ord=1)
        #if(desc_dist<0.15):
        if(self.tracked_count>5):
            self.xmin = self.pred_xmin
            self.ymin = self.pred_ymin
            self.xmax = self.pred_xmax
            self.ymax = self.pred_ymax
        elif(self.missed_count<3):
            self.xmin = self.pred_xmin
            self.ymin = self.pred_ymin
            self.xmax = self.pred_xmax
            self.ymax = self.pred_ymax
        
#             
        #elif(desc_dist<0.2):
            #self.xmin = (self.pred_xmin + self.xmin)/2
            #self.ymin = (self.pred_ymin+ self.ymin)/2
            #self.xmax = (self.pred_xmax+ self.xmax)/2
            #self.ymax = (self.pred_ymax+ self.ymax)/2
            
            #if(self.missed_count>6):
                #self.conf=0
        #else:
            
            #if(self.missed_count>6):
                #self.conf=0
            #if(self.missed_count>6):
                #self.conf =-0.1
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
        return "xmin: %f, ymin:%f, xmax:%f, ymax:%f, conf:%f, class:%d"%(self.xmin,self.ymin,self.xmax,self.ymax,self.conf,self.pred_class)
        
           
            
    