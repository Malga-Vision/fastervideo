import numpy as np
import cv2 as cv2
from .utils import *
class Detection(object):
    def __init__(self,conf,bbox,pred_class,desc=[]):
        self.xmin = float(bbox[0])
        self.ymin = float(bbox[1])
        self.xmax = float(bbox[2])
        self.ymax = float(bbox[3])
        self.conf = float(conf)
        self.pred_class = pred_class
        self.color=[]
        self.descriptor = np.array(list(map(float,desc)))
        
    def topleft(self):
        return np.array([self.xmin,self.ymin],np.float32)
    def topright(self):
        return np.array([self.xmax,self.ymin],np.float32)
    def botleft(self):
        return np.array([self.xmin,self.ymax],np.float32)
    def botright(self):
        return np.array([self.xmax,self.ymax],np.float32)
    def corners(self):
        z = np.zeros(4,np.float32)
        z[:2] = self.topleft()
        z[2:4] = self.botright()
        return z
    def center(self):
        return np.array([(self.xmin+self.xmax)/2,(self.ymin+self.ymax)/2],np.float32)
    def calc_hog_descriptor(self,frame):
       
        self.hog = get_hog_descriptor(frame,self.xmin,self.ymin,self.xmax,self.ymax)
        
    def calc_color_descriptor(self,frame):
        f = cv2.imread(frame)
        self.color = get_color_descriptor(f,self.xmin,self.ymin,self.xmax,self.ymax)
        
    def copy(self):
        other = Detection(self.conf,[self.xmin,self.ymin,self.xmax,self.ymax],self.pred_class)
        return other
    def area(self):
        return (self.xmax-self.xmin)*(self.ymax-self.ymin)
    def __repr__(self):
        return "xmin: %f, ymin:%f, xmax:%f, ymax:%f, conf:%f, class:%d"%(self.xmin,self.ymin,self.xmax,self.ymax,self.conf,self.pred_class)
  