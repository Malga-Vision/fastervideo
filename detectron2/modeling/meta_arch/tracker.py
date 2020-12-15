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
        self.history_window = 3
        self.detect_interval=3
        self.track_len = 10
        self.distances = []
        
    
    def get_distance_matrix(self,dets,tracks,frame):
        sup_count = 0
        dists = np.zeros((len(dets),len(tracks)),np.float32)
        for itrack in range(len(tracks)):
            for ipred in range(len(dets)):
                desc_dist=0
                iou_overlap = g_iou(dets[ipred].corners(),tracks[itrack].corners())
                if(self.use_appearance==True):
                    
                    if(self.embed==True or self.reid==True):
                    	if(iou_overlap < -0.25):
                    	    sup_count =+1
                    	    desc_dist = 999
                    	else:
                            if(self.dist=='cosine'):
                                desc_dist = distance.cosine(dets[ipred].descriptor,tracks[itrack].descriptor)/self.dist_thresh
                            
                        
                            else:
                                desc_dist = np.linalg.norm(dets[ipred].descriptor-tracks[itrack].descriptor, ord=2)/self.dist_thresh
                        
                    else:
                        
                        desc_dist = np.linalg.norm(dets[ipred].hog-tracks[itrack].hog, ord=1)/self.dist_thresh
                       
                        
                
                
                
                iou_dist = 1-iou_overlap
                
                self.distances.append([self.frameCount,dets[ipred].pred_class,tracks[itrack].pred_class,desc_dist,iou_dist,dets[ipred].corners(),tracks[itrack].corners()])
               
                
                total_dist = iou_dist +desc_dist 
            
                dists[ipred,itrack] = total_dist
               
        print('suppresed are',sup_count)
        return dists
    def get_predicted_tracks(self,frame_gray,prev_gray,scale_x,scale_y):
        adds = []
        for t,trk in enumerate(self.tracks):
            if(self.use_kalman==True):
                if(trk.tracked_count>5):

                    trk.apply_prediction(None,None)
                
            
            adds.append([trk.conf,trk.xmin,trk.ymin,trk.xmax,trk.ymax])
        return adds
    def filter_proposals(self,dets,frame_gray,prev_frame_gray):
       
        return [],[]

    def track(self,dets_org,descs_tensor ,frame,prev_gray,cur):
        frame_gray =None# cv.imread(frame, cv.COLOR_BGR2GRAY)
        
        dets_tensor = dets_org
        
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
            if(self.use_appearance==True and  self.hog ==False):
                dets.append(Detection(float(np.array((dets_tensor.scores[int(i)]),ndmin=1)[0]),[xmin,ymin,xmax,ymax],int(np.array(dets_tensor.pred_classes[int(i)],ndmin=1)),descs_tensor[i].numpy().ravel()))
            else:
                dets.append(Detection(float(np.array((dets_tensor.scores[int(i)]),ndmin=1)[0]),[xmin,ymin,xmax,ymax],int(np.array(dets_tensor.pred_classes[int(i)],ndmin=1)),np.array([])))
            
                
            if(self.use_appearance==True and self.hog==True):
                dets[i].calc_hog_descriptor(frame_gray,self.hog_num_cells)
                dets[i].calc_major_color(cur)
            else:
                dets[i].hog=None
            
        list_classes = [d.pred_class for d in dets]
        list_classes_tracks = [d.pred_class for d in self.tracks]
        list_classes = list(set(list_classes).union(set(list_classes_tracks)))
        
        for trk in self.tracks:
            trk.occluded  = False
        detect_occ = False
        if(detect_occ):
            for trk in self.tracks:
                for others in self.tracks:
                    if(trk.track_id==others.track_id or not (trk.pred_class == others.pred_class)):
                        continue
                    if(ios(trk.corners(),others.corners())>=1):
                        if(others.ymax>= trk.ymax):
                        
                            trk.occluded = True
                    
                        break
     

        for pred_class in list_classes:
            dets_class = [d for d in dets if d.pred_class == pred_class]
            track_class = [t for t in self.tracks if t.pred_class == pred_class]
            
            dists = self.get_distance_matrix(dets_class,track_class,frame_gray)
            
            r,c = linear_sum_assignment(dists)
            
            for ri,rval in enumerate(r):
                limit = 0.75
                if(self.use_appearance==True):
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
                    if(self.use_overlap):
                        for others in self.tracks:
                            if(trk.track_id==others.track_id):
                                continue
                            if(ios(trk.corners(),others.corners())>=self.overlap_threshold):
                                trk.is_overlap = True
                                break
                    trk.update(dets_class[np.where(c==t)[0][0]],frame_gray,prev_gray)
                    matched +=1
            for d,det in enumerate(dets_class):
                if(d not in r and det.conf>0.6):
                    possible_fp = False
                    for t,trk in enumerate(track_class):
                        if(ios(det.corners(),trk.corners())>self.fp_thresh or iou(det.corners(),trk.corners())>0.7):
                            possible_fp = True
                            break
                    if(self.suppress_fp==True):
                        if(possible_fp == True):
                           
                            continue
                    missed_dets+=1
                    
                    self.tracks.append(Track(self.tracking_method,self.cur_id,det,frame_gray,measurement_noise = self.measurement_noise,process_noise = self.process_noise, embed_alpha = self.embed_alpha))
                    self.cur_id+=1

        self.frameCount+=1
        
    
    def get_display_tracks(self):
        
    
        
        self.tracks = [track for track in self.tracks if track.conf>=0 and track.missed_count<self.track_life]
        
        res =  [track for track in self.tracks if track.conf>=0 and track.missed_count<self.track_visibility]
        
        return res
        
        
   
       

