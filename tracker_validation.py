import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random





#from google.colab.patches import cv2_imshow






# import some common detectron2 utilities

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.evaluation import COCOEvaluator,PascalVOCDetectionEvaluator
import matplotlib.pyplot as plt
import torch.tensor as tensor
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import inference_on_dataset
import torch
from detectron2.structures.instances import Instances
from detectron2.modeling import build_model
from detectron2.modeling.meta_arch.tracker import Tracker
from detectron2.modeling.meta_arch.soft_tracker import SoftTracker
#%matplotlib inline

cfg = get_cfg()
#cfg.MODEL.DEVICE='cpu'
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x_Video.yaml")
#cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_50_Video.yaml")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # set threshold for this model
#cfg.MODEL.WEIGHTS = "KITTI_FPN_FINAL/model_final.pth"
cfg.MODEL.WEIGHTS = "../models/kitti_jde.pth"

print(cfg.MODEL)
#arr = {1:'cyclist',2:'car',0:'pedestrian'}
arr = {2:'Car',0:'Pedestrian'}
print(arr)
from contextlib import contextmanager
import sys, os
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
def print_val_results(results_name):

    with suppress_stdout():
        print("Now you don't")
        os.system('../../python2 devkit_tracking/python/validate_tracking.py val')
    
    
    labels = {1:'MOTA',2:'MOTP',3:'MOTAL',4:'MODA',5:'MODP',7:'R',8:'P',12:'MT',13:'PT',14:'ML',18:'FP',19:'FN',22:'IDs'}
    summary_heading = 'Metric\t'
    for label in labels.keys():
        summary_heading+=labels[label] + '\t'
    summary_cars = 'Cars\t'
    summary_peds = 'Peds\t'
    with open('../../devkit_tracking/python/results/'+results_name+'/summary_car.txt') as f:
        i=0
        for line in f:
            if(i==0):
                i+=1
                continue
            if(i in labels.keys()):
                summary_cars+= str(round(float(line[len(line)-9:len(line)-1].strip()),2))+'\t'
            i+=1


    with open('../devkit_tracking/python/results/'+results_name+'/summary_pedestrian.txt') as f:
        i=0
        for line in f:
            if(i==0):
                i+=1
                continue
            if(i in labels.keys()):
                summary_peds+= str(round(float(line[len(line)-9:len(line)-1].strip()),2))+'\t'
            i+=1
    print(summary_heading)
    print(summary_cars)
    print(summary_peds)
def print_test_results(results_name):

    with suppress_stdout():
        print("Now you don't")
        os.system('python2 ../devkit_tracking/python/evaluate_tracking.py test')
    
    
    labels = {1:'MOTA',2:'MOTP',3:'MOTAL',4:'MODA',5:'MODP',7:'R',8:'P',12:'MT',13:'PT',14:'ML',18:'FP',19:'FN',22:'IDs'}
    summary_heading = 'Metric\t'
    for label in labels.keys():
        summary_heading+=labels[label] + '\t'
    summary_cars = 'Cars\t'
    summary_peds = 'Peds\t'
    with open('../../devkit_tracking/python/results/'+results_name+'/summary_car.txt') as f:
        i=0
        for line in f:
            if(i==0):
                i+=1
                continue
            if(i in labels.keys()):
                summary_cars+= str(round(float(line[len(line)-9:len(line)-1].strip()),2))+'\t'
            i+=1


    with open('../../devkit_tracking/python/results/'+results_name+'/summary_pedestrian.txt') as f:
        i=0
        for line in f:
            if(i==0):
                i+=1
                continue
            if(i in labels.keys()):
                summary_peds+= str(round(float(line[len(line)-9:len(line)-1].strip()),2))+'\t'
            i+=1
    print(summary_heading)
    print(summary_cars)
    print(summary_peds)
def print_debug_results(results_name):

    with suppress_stdout():
        print("Now you don't")
        os.system('python2 ../devkit_tracking/python/debug_tracking.py debug')
    
    
    labels = {1:'MOTA',2:'MOTP',3:'MOTAL',4:'MODA',5:'MODP',7:'R',8:'P',12:'MT',13:'PT',14:'ML',18:'FP',19:'FN',22:'IDs'}
    summary_heading = 'Metric\t'
    for label in labels.keys():
        summary_heading+=labels[label] + '\t'
    summary_cars = 'Cars\t'
    summary_peds = 'Peds\t'
    with open('../../devkit_tracking/python/results/'+results_name+'/summary_car.txt') as f:
        i=0
        for line in f:
            if(i==0):
                i+=1
                continue
            if(i in labels.keys()):
                summary_cars+= str(round(float(line[len(line)-9:len(line)-1].strip()),2))+'\t'
            i+=1


    with open('../../devkit_tracking/python/results/'+results_name+'/summary_pedestrian.txt') as f:
        i=0
        for line in f:
            if(i==0):
                i+=1
                continue
            if(i in labels.keys()):
                summary_peds+= str(round(float(line[len(line)-9:len(line)-1].strip()),2))+'\t'
            i+=1
    print(summary_heading)
    print(summary_cars)
    print(summary_peds)
import json
import os
import time
from tqdm.notebook import tqdm
colors = [[0,0,128],[0,255,0],[0,0,255],[255,0,0],[0,128,128],[128,0,128],[128,128,0],[255,255,0],[0,255,255],[255,255,0],[128,0,0],[0,128,0]
         ,[0,128,255],[0,255,128],[255,0,128],[128,255,0],[255,128,0],[128,255,255],[128,0,255],[128,128,128],[128,255,128]]
dirC = '/media/DATA/Datasets/KITTI/tracking/data_tracking_image_2/training/image_02/'
names = []





output_path = '../../devkit_tracking/python/results/'
settings = [
   
    #dict(props=20,st=1.15,T=True,D='cosine',Re=True,A=True,H=False,K=True,E=False,O=False,overlap_threshold = 0.6,measurement=0.001,process=1,hog_cells=4,dist_thresh=1.5,track_life=7,track_vis=2),
    
    #dict(props=20,st=1.05,T=True,D='cosine',Re=True,A=True,H=False,K=True,E=False,O=False,overlap_threshold = 0.6,measurement=0.001,process=1,hog_cells=4,dist_thresh=1.5,track_life=7,track_vis=2),
    
    #dict(props=20,st=1.05,an=3,T=True,D='cosine',Re=True,A=True,H=False,K=True,E=False,O=False,overlap_threshold = 0.6,measurement=0.001,process=1,hog_cells=4,dist_thresh=1.5,track_life=7,track_vis=2),
    dict(props=20,st=1.1,an=3,T=True,D='cosine',Re=True,A=True,H=False,K=True,E=False,O=False,overlap_threshold = 0.6,measurement=0.001,process=1,hog_cells=4,dist_thresh=1.5,track_life=7,track_vis=2),
    #dict(props=20,st=1.15,an=3,T=True,D='cosine',Re=True,A=True,H=False,K=True,E=False,O=False,overlap_threshold = 0.6,measurement=0.001,process=1,hog_cells=4,dist_thresh=1.5,track_life=7,track_vis=2),
    #dict(props=20,st=1.2,an=3,T=True,D='cosine',Re=True,A=True,H=False,K=True,E=False,O=False,overlap_threshold = 0.6,measurement=0.001,process=1,hog_cells=4,dist_thresh=1.5,track_life=7,track_vis=2),
    #dict(props=20,st=1.25,an=3,T=True,D='cosine',Re=True,A=True,H=False,K=True,E=False,O=False,overlap_threshold = 0.6,measurement=0.001,process=1,hog_cells=4,dist_thresh=1.5,track_life=7,track_vis=2),
    #dict(props=20,st=1.3,an=3,T=True,D='cosine',Re=True,A=True,H=False,K=True,E=False,O=False,overlap_threshold = 0.6,measurement=0.001,process=1,hog_cells=4,dist_thresh=1.5,track_life=7,track_vis=2),
    
    #dict(props=20,T=False,D='cosine',Re=True,A=True,H=False,K=True,E=False,O=False,overlap_threshold = 0.6,measurement=0.001,process=1,hog_cells=4,dist_thresh=1.5,track_life=7,track_vis=2),
    #dict(props=20,T=True,D='cosine',Re=False,A=True,H=False,K=True,E=True,O=False,overlap_threshold = 0.6,measurement=0.001,process=1,hog_cells=4,dist_thresh=1.5,track_life=7,track_vis=2),
    #dict(props=20,T=False,D='cosine',Re=False,A=True,H=False,K=True,E=True,O=False,overlap_threshold = 0.6,measurement=0.001,process=1,hog_cells=4,dist_thresh=1.5,track_life=7,track_vis=2),
    #dict(props=20,T=True,D='cosine',Re=False,A=True,H=True,K=True,E=False,O=False,overlap_threshold = 0.6,measurement=0.001,process=1,hog_cells=4,dist_thresh=15,track_life=7,track_vis=2),
    #dict(props=20,T=False,D='cosine',Re=False,A=False,H=False,K=True,E=False,O=False,overlap_threshold = 0.6,measurement=0.001,process=1,hog_cells=4,dist_thresh=1.5,track_life=7,track_vis=2),
     
  
    
    
]
train_folders = ['0000','0002','0003','0004','0005','0009','0011','0017','0020']
val_folders = ['0001','0006','0008','0016','0018','0019']
test_folders_2 = ['0002','0007','0009','0013','0016','0017','0018','0019']
test_folders = ['0014','0015','0016','0018','0019','0001','0006','0008','0010','0012','0013']
debug_folders = ['0002']
submission_folders = ['0000','0001','0002','0003','0004','0005','0006','0007',
                      '0008','0009','0010','0011','0012','0013','0014','0015','0016','0017',
                      '0018','0019','0020','0021','0022','0023','0024','0025','0026','0027','0028']
final_test_folders = ['0014']
for setting in settings:
    test_name = 'debug'
    exp_name = output_path+ '/'+test_name
    
    if(not os.path.exists(exp_name)):
      os.mkdir(exp_name)
      os.mkdir(exp_name+'/data')
    
    for folder_name in debug_folders:
      #out_tracking = cv2.VideoWriter('joint_%s.avi'%folder_name,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1242,375))
      
      #if(not folder_name in ['0000','0007','0011']):
        #continue
      dump_image = cv2.imread(dirC+folder_name+'/000001.png')
      out_tracking = cv2.VideoWriter('debug_kitti.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, dump_image.shape[1::-1])
      predictor = DefaultPredictor(cfg)
      #prop_limit=60
      predictor.model.tracker = SoftTracker()
      print(folder_name)
      predictor.model.tracking_proposals = setting['T']
      predictor.model.tracker.track_life = setting['track_life']
      predictor.model.tracker.track_visibility = setting['track_vis']
      predictor.model.tracker.use_appearance = setting['A']
      predictor.model.tracker.use_kalman = setting['K']
      predictor.model.tracker.use_overlap = setting['O']
      predictor.model.tracker.hot = setting['H']
      predictor.model.tracker.use_color = False
      predictor.model.tracker.embed = setting['E']
      predictor.model.tracker.reid = setting['Re']
      predictor.model.tracker.hog = setting['H']
      predictor.model.tracker.dist = setting['D']
      predictor.model.tracker.measurement_noise=setting['measurement']
      predictor.model.tracker.process_noise = setting['process']
      predictor.model.enable_clustering=False
      predictor.model.tracker.hog_num_cells = setting['hog_cells']
      predictor.model.tracker.dist_thresh = setting['dist_thresh']
      predictor.model.tracker.overlap_threshold = setting['overlap_threshold']
      predictor.model.use_reid = setting['Re']
      predictor.model.tracker.angle_norm = setting['an']
      predictor.model.tracker.soft_thresh = setting['st']
      max_distance = 0.2
      
      
      
        
      output_file = open('%s/data/%s.txt'%(exp_name,folder_name),'w')
      
      start = time.time()
      frame_counter = 0
      prev_path = 0
      predictor.model.prev_path = 0
      for photo_name in sorted(os.listdir(dirC+folder_name+'/')):
        img_path = dirC+folder_name+'/'+photo_name
        #print(img_path)
        img = cv2.imread(img_path)
        inp = {}
        inp['width'] = img.shape[1]
        inp['height'] = img.shape[0]



        inp['file_name'] =  photo_name
        inp['image_id'] = photo_name
        predictor.model.remove_duplicate_tracks=False
        predictor.model.photo_name = img_path
        #outputs = predictor.model([img])
        outputs = predictor(img,setting['props'],max_distance)
        
        #res = outputs["instances"].to("cpu")

       
        for i in outputs:


          #tracking_res.append({"image_id" : int(name), "category_id" : int(np.array(res.pred_classes[int(i)],ndmin=1))+1, "bbox" : [int(xmin),int(ymin),int(xmax-xmin),int(ymax-ymin)],"id":gl_id, "score" :float(np.array((res.scores[int(i)]),ndmin=1)[0])})# np.minimum(1.0,np.maximum(box.conf,0.5))})

          
          if(i.pred_class in arr):
              color = colors[i.track_id%len(colors)]
              
                
              cv2.rectangle(img, (int(i.xmin), int(i.ymin)), (int(i.xmax),int(i.ymax)),color, 3)
              #cv2.putText(img,arr[i.pred_class], (int(i.xmin), int(i.ymin)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
              output_file.write("%d %d %s 0.00 0 -0.20 %d %d %d %d 1.89 0.48 1.20 1.84 1.47 8.41 0.01 %f\n"%(frame_counter,i.track_id,arr[i.pred_class],i.xmin,i.ymin,i.xmax,i.ymax,i.conf))
        #cv2.imwrite('kitti_2_debug//%s'%(photo_name),img)
        out_tracking.write(img)
        #cv2.imwrite('KITTI_results/%s.jpg'%name,img)
        
        frame_counter +=1
        predictor.model.prev_path = img_path
      plt.hist(predictor.model.tracker.cand_det)
      print('above 1 are %d'%len(np.where(np.array(predictor.model.tracker.cand_det)>1)[0]))
      plt.savefig('hist.png')
      plt.show()
      end = time.time()
      elapsed = end-start
      out_tracking.release()
      avg = frame_counter/elapsed
      print('avg time is' ,avg)
      output_file.close()
      
    print(setting) 
    print_debug_results(test_name)
    #HOG
    plt.figure()
    #print(np.max([p[3] for p in predictor.model.tracker.distances]))
    #plt.scatter([p[3] for p in predictor.model.tracker.distances], [p[4] for p in predictor.model.tracker.distances] ,s=1)
    
          