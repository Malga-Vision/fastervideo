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


class SoftTracker(object):
    def __init__(self,method='kalman_vel'):
        self.tracking_method = method
        
        self.tracks = []
        self.cur_id=1
        self.frameCount =1
        self.detect_interval=3
        self.track_len = 10
        self.distances = []
        self.cand_det = []
        self.feature_params=dict(maxCorners=200,qualityLevel=0.3,minDistance=7,blockSize=7)
        self.lk_params=dict(winSize=(15,15),maxLevel=2,criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT,10,0.03))
        self.flow_time=0
    def get_iou_distance_matrix(self,dets,tracks,frame):
        dists = np.zeros((len(dets),len(tracks)),np.float32)
        for itrack in range(len(tracks)):
            for ipred in range(len(dets)):
                desc_dist=0
                if(self.use_appearance==True):
                    
                    if(self.embed==True or self.reid==True):
                        if(self.dist=='cosine'):
                            desc_dist = distance.cosine(dets[ipred].descriptor,tracks[itrack].descriptor)/self.dist_thresh
                           
                        else:
                            desc_dist = np.linalg.norm(dets[ipred].descriptor-tracks[itrack].descriptor, ord=2)/self.dist_thresh
                       
                    else:
                        
                        desc_dist = np.linalg.norm(dets[ipred].hog-tracks[itrack].hog, ord=1)/self.dist_thresh
                       
                        
                
                iou_overlap = iou(dets[ipred].corners(),tracks[itrack].corners())
                iou_dist = 1-iou_overlap
                
                self.distances.append([self.frameCount,dets[ipred].pred_class,tracks[itrack].pred_class,desc_dist,iou_dist,dets[ipred].corners(),tracks[itrack].corners()])
                
                total_dist = iou_dist 
            
                dists[ipred,itrack] = total_dist
                
        return dists
    def get_distance_matrix(self,dets,tracks,frame):
        
        dists = np.zeros((len(dets),len(tracks)),np.float32)
        for itrack in range(len(tracks)):
            for ipred in range(len(dets)):
                desc_dist=0
                if(self.use_appearance==True):
                    
                    if(self.embed==True or self.reid==True):
                        if(self.dist=='cosine'):
                            desc_dist = distance.cosine(dets[ipred].descriptor,tracks[itrack].descriptor)/self.dist_thresh
                           
                        else:
                            desc_dist = np.linalg.norm(dets[ipred].descriptor-tracks[itrack].descriptor, ord=2)/self.dist_thresh
                       
                    else:
                        
                        desc_dist = np.linalg.norm(dets[ipred].hog-tracks[itrack].hog, ord=1)/self.dist_thresh
                       
                        
                iou_overlap = iou(dets[ipred].corners(),tracks[itrack].corners())
                iou_dist = 1-iou_overlap
                
                self.distances.append([self.frameCount,dets[ipred].pred_class,tracks[itrack].pred_class,desc_dist,iou_dist,dets[ipred].corners(),tracks[itrack].corners()])
                
                total_dist = iou_dist +desc_dist
            
                dists[ipred,itrack] = total_dist
            
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
            
            dets.append(Detection(float(np.array((dets_tensor.scores[int(i)]),ndmin=1)[0]),[xmin,ymin,xmax,ymax],int(np.array(dets_tensor.pred_classes[int(i)],ndmin=1)),descs_tensor[i].numpy().ravel()))
                
            
                
            if(self.use_appearance==True and self.hog==True):
                dets[i].calc_hog_descriptor(frame_gray,self.hog_num_cells)
                dets[i].calc_major_color(cur)
            else:
                dets[i].hog=None
            
        list_classes = [d.pred_class for d in dets]
        list_classes_tracks = [d.pred_class for d in self.tracks]
        list_classes = list(set(list_classes).union(set(list_classes_tracks)))
        
        for pred_class in list_classes:
            dets_class = [d for d in dets if d.pred_class == pred_class]
            track_class = [t for t in self.tracks if t.pred_class == pred_class]
            
            dists = self.get_distance_matrix(dets_class,track_class,frame_gray)
            dists_iou = self.get_iou_distance_matrix(dets_class,track_class,frame_gray)
            det_indices,track_indices = linear_sum_assignment(dists)
            global_used_dets = []
            
            for ri,rval in enumerate(det_indices):
                limit = 1.2
                if(self.use_appearance==True):
                    limit  =1.2
                if(dists[det_indices[ri],track_indices[ri]]>limit):
                 
                      
                  det_indices[ri] = -1
                  track_indices[ri] = -1
            for t,trk in enumerate(track_class):
                
                if(t not in track_indices ):
                    
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
                    
                    det_index = np.where(track_indices==t)[0][0]
                    dist_min_iou = dists_iou[det_index,t]
                    if(dist_min_iou>0.99):
                        continue
                    dist_min  =dists[det_index,t]
                    candidate_detections = [] 
                    weights  =[]
                    for c_id,d in enumerate(dists[:,t]):
    
                        if(d/dist_min >=0.99 and d/dist_min<self.soft_thresh and dists_iou[c_id,t]<1):
                            candidate_detections.append(c_id)
                            weights.append(1/d)
                            global_used_dets.append(c_id)
                    if(len(candidate_detections)>0):
                       
                        self.cand_det.append(len(candidate_detections))
                        avg_conf = np.average([dets_class[c_id].conf for c_id in candidate_detections], weights = weights)
                        avg_descriptor = np.average([dets_class[c_id].descriptor for c_id in candidate_detections],axis=0,weights = weights)
                        avg_xmin = np.average([dets_class[c_id].xmin for c_id in candidate_detections],weights = weights)
                        avg_ymin = np.average([dets_class[c_id].ymin for c_id in candidate_detections],weights = weights)
                        avg_xmax = np.average([dets_class[c_id].xmax for c_id in candidate_detections],weights = weights)
                        avg_ymax = np.average([dets_class[c_id].ymax for c_id in candidate_detections],weights = weights)
                        
                        trk.update(Detection(avg_conf,[avg_xmin,avg_ymin,avg_xmax,avg_ymax],pred_class,avg_descriptor),frame_gray,prev_gray)
                        matched +=1
            for d,det in enumerate(dets_class):
                if(not (d in det_indices or d in global_used_dets) and  det.conf>0.6):
                    missed_dets+=1
                    possible_fp = False
                    for t,trk in enumerate(track_class):
                        if(ios(det.corners(),trk.corners())>self.fp_thresh or iou(det.corners(),trk.corners())>0.7):
                            possible_fp = True
                            break
                    if(self.suppress_fp==True):
                        if(possible_fp == True):
                            
                            continue
                    self.tracks.append(Track(self.tracking_method,self.cur_id,det,frame_gray,measurement_noise = self.measurement_noise,process_noise = self.process_noise,embed_alpha = self.embed_alpha))
                    self.cur_id+=1

        self.frameCount+=1
        
    
    def get_display_tracks(self):
        
    
        
        self.tracks = [track for track in self.tracks if track.conf>=0 and track.missed_count<self.track_life]

        res =  [track for track in self.tracks if track.conf>=0 and track.missed_count<self.track_visibility]
        
        return res
        
        
   
       

