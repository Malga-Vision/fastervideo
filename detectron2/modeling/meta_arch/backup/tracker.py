import numpy as np
import math
import scipy.interpolate as interp
from scipy.spatial import distance
import imutils
import time
from sklearn import preprocessing
import cv2 as cv
import time
from scipy.optimize import linear_sum_assignment

from .utils import *
from detectron2.structures import Instances
from detectron2.structures import Boxes
from .detection import Detection
from .track import Track
import torch


class Tracker(object):
    def __init__(self,method='kalman_acc'):
        self.tracking_method = method
        
        self.tracks = []
        self.cur_id=1
        self.frameCount =1
        self.detect_interval=3
        self.track_len = 10
        self.distances = []
        self.feature_params=dict(maxCorners=200,qualityLevel=0.3,minDistance=7,blockSize=7)
        self.lk_params=dict(winSize=(15,15),maxLevel=2,criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT,10,0.03))
        self.flow_time=0
    
    def get_distance_matrix(self,dets,tracks,frame):
        
        dists = np.zeros((len(dets),len(tracks)),np.float32)
        for itrack in range(len(tracks)):
            for ipred in range(len(dets)):
                desc_dist=0
                if(self.use_appearance==True):
                    if(self.use_embed==True):
                        desc_dist = distance.cosine(dets[ipred].descriptor,self.tracks[itrack].descriptor)
                        
                    else:
                        
                        desc_dist = np.linalg.norm(dets[ipred].hog-self.tracks[itrack].hog, ord=1)/1.2
                       
                        #desc_dist = np.linalg.norm(np.array(dets[ipred].major_color)-np.array(self.tracks[itrack].major_color))/2
                
                iou_overlap = iou(dets[ipred].corners(),tracks[itrack].corners())
                iou_dist = 1-iou_overlap
                self.distances.append([self.frameCount,dets[ipred].pred_class,tracks[itrack].pred_class,desc_dist,iou_dist,dets[ipred].corners(),tracks[itrack].corners()])
                #uncertainety =np.maximum(1-dets[ipred].conf,0.5)
                total_dist = iou_dist +desc_dist
            
                #if(desc_dist>8):
                    #total_dist = total_dist +1.0
                dists[ipred,itrack] = total_dist
                #print(iou_dist,desc_dist)
                
                #dists[ipred,itrack] = ((1-iou_overlap)+desc_dist)
        return dists
    def get_predicted_tracks(self,frame_gray,prev_gray,scale_x,scale_y):
        adds = []
        for t,trk in enumerate(self.tracks):
            if(self.use_kalman==True):
                if(trk.tracked_count>5):
                    #print('happened')
                    trk.apply_prediction(None,None)
                
            #adds.append([trk.conf,trk.pred_xmin*0.9323,trk.pred_ymin*0.9310,trk.pred_xmax*0.9323,trk.pred_ymax*0.9310])
            adds.append([trk.conf,trk.xmin,trk.ymin,trk.xmax,trk.ymax])
        return adds
    def filter_proposals(self,dets,frame_gray,prev_frame_gray):
        dists = self.get_simple_distance_matrix(dets,[t for t in self.tracks ] ,frame_gray)
        
        r,c = linear_sum_assignment(dists)
        for ri,rval in enumerate(r):
          
          if(dists[r[ri],c[ri]]>0.8):
            r[ri] = -1
            c[ri] = -1

        for trk in self.tracks:
            trk.matched=False
        adds = []
        inds = []
        for d,det in enumerate(dets):
            inds.append(d)
            #if(d not in r ):
                #inds.append(d)
        
        for t,trk in enumerate(self.tracks):
            if(t not in c):
              
                trk.apply_prediction(None,None)
                
                adds.append([trk.conf,trk.pred_xmin,trk.pred_ymin,trk.pred_xmax,trk.pred_ymax])
            else:
                
                trk.update(dets[np.where(c==t)[0][0]],None,None)
        return inds,adds

    def track(self,dets_org,descs_tensor ,frame,prev_gray,cur):
        frame_gray = cv.imread(frame, cv.COLOR_BGR2GRAY)
        dets_tensor = dets_org
        descs_tensor = descs_tensor.to('cpu')
        self.image_size = dets_tensor._image_size
        dets = []
        missed_tracks = 0
        missed_dets = 0
        matched = 0
        for i in np.arange(len(dets_tensor.pred_boxes)):
            
            xmin = dets_tensor.pred_boxes[int(i)].tensor[0,0].numpy()
            ymin = dets_tensor.pred_boxes[int(i)].tensor[0,1].numpy()
            xmax = dets_tensor.pred_boxes[int(i)].tensor[0,2].numpy()
            ymax = dets_tensor.pred_boxes[int(i)].tensor[0,3].numpy()
            
            dets.append(Detection(float(np.array((dets_tensor.scores[int(i)]),ndmin=1)[0]),[xmin,ymin,xmax,ymax],int(np.array(dets_tensor.pred_classes[int(i)],ndmin=1)),descs_tensor[i]))
                
            
                
            if(self.use_appearance==True and self.use_embed==False):
                dets[i].calc_hog_descriptor(frame_gray)
                dets[i].calc_major_color(cur)
            else:
                dets[i].hog=None
            
        list_classes = [d.pred_class for d in dets]
        list_classes_tracks = [d.pred_class for d in self.tracks]
        list_classes = list(set(list_classes).union(set(list_classes_tracks)))
        #print(list_classes)
        for pred_class in [2,0]:
            dets_class = [d for d in dets if d.pred_class == pred_class]
            track_class = [t for t in self.tracks if t.pred_class == pred_class]
            
            dists = self.get_distance_matrix(dets_class,track_class,frame_gray)
            
            r,c = linear_sum_assignment(dists)
            
            for ri,rval in enumerate(r):
                limit = 0.4
                if(self.use_appearance==True):
                  if(self.use_embed==True):
                    limit = 1.2
                  else:
                    limit  =1.2
                if(dists[r[ri],c[ri]]>limit):
                 
                      
                  r[ri] = -1
                  c[ri] = -1
             
            for t,trk in enumerate(track_class):
                
                if(t not in c ):
                    
                    trk.matched = False
                    trk.missed_count+=1
                    trk.tracked_count=0 
                    
                    trk.apply_prediction(frame_gray,prev_gray)
                    missed_tracks+=1
                    
                else:
                    
                    
                    trk.is_overlap = False
                    for others in self.tracks:
                        if(trk.track_id==others.track_id):
                            continue
                        if(ios(trk.corners(),others.corners())>=0.3):
                            trk.is_overlap = True
                            break
                    trk.update(dets_class[np.where(c==t)[0][0]],frame_gray,prev_gray)
                    matched +=1
            for d,det in enumerate(dets_class):
                if(d not in r and det.conf>0.6):
                    missed_dets+=1
                    
                    self.tracks.append(Track(self.tracking_method,self.cur_id,det,frame_gray))
                    self.cur_id+=1

        self.frameCount+=1
        
    
    def get_display_tracks(self):
        #for t in self.tracks:
            #t.predict(None,None)
       
    
        
        self.tracks = [track for track in self.tracks if track.conf>=0 and track.missed_count<7]

        res =  [track for track in self.tracks if track.conf>=0 and track.missed_count<2]
        
        return res
        
        
   
       

