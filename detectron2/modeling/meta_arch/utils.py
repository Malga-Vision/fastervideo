from .detection import *
import numpy as np
import cv2 as cv2
from scipy.spatial import distance
from skimage.feature import hog
import scipy.interpolate as interp

def get_overlap_to_self(a,b):
    if(int(a.xmin)==int(b.xmin) and int(a.ymin)==int(b.ymin) and int(a.xmax)==int(b.xmax) and int(a.ymax)==int(b.ymax)):
        return 0
    x1 = max(a.xmin, b.xmin)
    y1 = max(a.ymin, b.ymin)
    x2 = min(a.xmax, b.xmax)
    y2 = min(a.ymax, b.ymax)

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height
    self_area = (a.xmax-a.xmin)*(a.ymax-a.ymin)
    if(self_area>0):
        res = area_overlap/self_area
    else:
        res = 0
  
    return res
def bounding_box_naive(points):
    """returns a list containing the bottom left and the top right 
    points in the sequence
    Here, we use min and max four times over the collection of points
    """
    

    top_left_x = min(point[0][0] for point in points)
    top_left_y = min(point[0][1] for point in points)
    bot_right_x = max(point[0][0] for point in points)
    bot_right_y = max(point[0][1] for point in points)

    center_x = (top_left_x+bot_right_x)/2
    center_y = (top_left_y+bot_right_y)/2
    return [center_x,center_y,bot_right_x-top_left_x,bot_right_y-top_left_y]
def bounding_box(points):
    """returns a list containing the bottom left and the top right 
    points in the sequence
    Here, we use min and max four times over the collection of points
    """
    
    
    top_left_x = min(point[0] for point in points)
    top_left_y = min(point[1] for point in points)
    bot_right_x = max(point[0] for point in points)
    bot_right_y = max(point[1] for point in points)

    center_x = (top_left_x+bot_right_x)/2
    center_y = (top_left_y+bot_right_y)/2
    return [center_x,center_y,bot_right_x-top_left_x,bot_right_y-top_left_y]
def load_detections(dataset,detector,boat_class,min_conf):
    text_file_path = "detections_no_desc/%s/%s.txt"%(dataset,detector)
    f = open(text_file_path,"r")
    line = f.readline()
    detections={}
    comps = []
    while(line):

        line = line.replace("\n", "")
        comps = line.split(",")
        
        if(int(comps[2])==boat_class and float(comps[3])>min_conf):
            
            if(not comps[0] in detections):
                detections[comps[0]]=[]
            if (not (int(comps[4])>270 and int(comps[4])<740 and int(comps[5])>540 and int(comps[6])>580)):
                
                detections[comps[0]].append(Detection(comps[3],comps[4:8],comps[8:]))
            
        line=f.readline()
        
    f.close()
    detections_after={}
    for k in detections.keys():
        cur= detections[k]
        detections_after[k] = []
        #print('there was ',len(cur),' detections')
        for item in cur:
            contained = False
            for item2 in cur:
                overlap = get_overlap_to_self(item,item2)
                
                if(overlap>0.5):
                    #print('overlap is ',overlap,' discarding...')
                    contained=True
                    break
            if(contained==False):
                detections_after[k].append(item.copy())
        #print('now there are ', len(detections_after[k]),' detections')
    return detections_after
def iou(a, b):
	epsilon=1e-5

	x1 = max(a[0], b[0])
	y1 = max(a[1], b[1])
	x2 = min(a[2], b[2])
	y2 = min(a[3], b[3])

	# AREA OF OVERLAP - Area where the boxes intersect
	width = (x2 - x1)
	height = (y2 - y1)
	# handle case where there is NO overlap
	if (width<0) or (height <0):
		return 0.0
	area_overlap = width * height

	# COMBINED AREA
	area_a = (a[2] - a[0]) * (a[3] - a[1])
	area_b = (b[2] - b[0]) * (b[3] - b[1])
	area_combined = area_a + area_b - area_overlap

	# RATIO OF AREA OF OVERLAP OVER COMBINED AREA
	iou = area_overlap / (area_combined+epsilon)
	return iou   
def g_iou(b_p,b_g):
	epsilon = 1e-5
	A_g = (b_g[2]-b_g[0])*(b_g[3]-b_g[1])
	A_p = (b_p[2]-b_p[0])*(b_p[3]-b_p[1])
	
	x1_i = max(b_p[0],b_g[0])
	y1_i = max(b_p[1],b_g[1])
	x2_i = min(b_p[2],b_g[2])
	y2_i = min(b_p[2],b_g[2])
	
	I = (x2_i - x1_i)* ( y2_i - y1_i)
	
	x1_c = min(b_p[0],b_g[0])
	y1_c = min(b_p[1],b_g[1])
	x2_c = max(b_p[2],b_g[2])
	y2_c = max(b_p[2],b_g[2])
	
	A_c = (x2_c - x1_c) * ( y2_c - y1_c)
	
	U = A_p + A_g - I
	
	IoU = I/U
	
	GIoU = IoU - ((A_c - U)/A_c)
	
	return GIoU
	
def ios(a,b):
	epsilon=1e-5

	x1 = max(a[0], b[0])
	y1 = max(a[1], b[1])
	x2 = min(a[2], b[2])
	y2 = min(a[3], b[3])

	# AREA OF OVERLAP - Area where the boxes intersect
	width = (x2 - x1)
	height = (y2 - y1)
	# handle case where there is NO overlap
	if (width<0) or (height <0):
		return 0.0
	area_overlap = width * height

	# COMBINED AREA
	area_a = (a[2] - a[0]) * (a[3] - a[1])
	
	area_combined = area_a  - area_overlap

	# RATIO OF AREA OF OVERLAP OVER COMBINED AREA
	iou = area_overlap / (area_combined+epsilon)
	return iou

def get_hog_descriptor(frame,xmin,ymin,xmax,ymax,num_cells=1):
    
    if(xmin<0):
        xmin = 0
    if(ymin<0):
        ymin=0
    if(xmax<0):
        xmax=0
    if(ymax<0):
        ymax=0
    
    section = frame[int(ymin):int(ymax),int(xmin):int(xmax)]
    
    if(section.shape[0]==0 or section.shape[1]==0):
       
        return np.zeros(9*num_cells*num_cells)
        
        #return np.zeros(36)
    #return hog(section,pixels_per_cell=(section.shape[0]/2,section.shape[1]/2),cells_per_block=(1, 1),feature_vector=True)
    return hog(section,pixels_per_cell=(section.shape[0]/num_cells,section.shape[1]/num_cells),cells_per_block=(num_cells, num_cells),feature_vector=True)
def get_points_in_bb(points,corners):
    sels = []

    for p in points:
        if p[0]>= corners[0] and p[0]<= corners[2] and p[1] >= corners[1] and p[1] <=corners[3]:
            sels.append(p)
    
    return sels

def get_distance(v1,v2):
    
    if(len(v1)==len(v2)):
        dist = distance.euclidean(np.array(v1),np.array(v2))
    elif(len(v1)>len(v2)):
        print('interpolation')
        v2_func = interp.interp1d(np.arange(len(v2)),v2)
        v2_mod = v2_func(np.linspace(0,len(v2)-1,len(v1)))
        dist = np.linalg.norm(np.array(v1)-np.array(v2_mod))
    else:
        print('interpolation')
        v1_func = interp.interp1d(np.arange(len(v1)),v1)
        v1_mod = v1_func(np.linspace(0,len(v1)-1,len(v2)))
        dist = np.linalg.norm(np.array(v1_mod)-np.array(v2))
    if(dist>1000):
        dist=30
    return dist
def calc_major_color(frame,xmin,ymin,xmax,ymax):
    #hsv = cv2.cvtColor(frame[int(ymin):int(ymax),int(xmin):int(xmax),:], cv2.COLOR_BGR2HSV)
    #v = np.bincount(hsv.ravel())
    lab = cv2.cvtColor(frame[int(ymin):int(ymax),int(xmin):int(xmax),:], cv2.COLOR_BGR2Lab)
    return [np.median(lab)]

    


