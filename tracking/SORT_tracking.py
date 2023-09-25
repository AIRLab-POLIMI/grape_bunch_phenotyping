import os 
import json
import collections
from pprint import pprint
from sort import *

#perform tracking on custom dataset, then save the image with the id found by the algorithm for each frame
KalmanBoxTracker.count=0
mot_tracker_def = Sort()#max_age=1,min_hits=3, iou_threshold=0.3)

with open(jsonpath) as data_file:    
   data = json.load(data_file)
odata = collections.OrderedDict(sorted(data.items()))

##COUNTING
ids_dict = {}

for key in odata.keys():
    print(key)
    arrlist = []
    #image_name = key.split("/")[3]
    #det_img = cv2.imread(os.path.join(img_path, image_name))
    #overlay = det_img.copy()
    det_result = data[key] 
    
    for info in det_result:
        bbox = info['bbox']
        #print(bbox)
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[0] + bbox[2]
        ymax = bbox[1] + bbox[3]
        new_bbox = [xmin, ymin, xmax, ymax]
        #print(new_bbox)
        
        #labels = info['category_id']
        scores = info['score']
        templist = new_bbox+[float(scores)]
        
        #if labels == 0:
        arrlist.append(templist)

           
    track_bbs_ids = mot_tracker_def.update(np.array(arrlist))
    
    ###################################################################################
    #COUNT FROM TRACKING:
    #here insert smth that keeps track of the count of different IDs in the frames and count the ones which appear after a certain treshold
    #e.g ID 1 appear in 40 frames, ID 68 appear in 5 frames, count IDs appearing over 30 frames
    #NB they should be smth like -64-
    for elem in track_bbs_ids:
        id_bbox = int(elem[-1])
        #print(elem)
        if id_bbox in ids_dict:
            value = ids_dict[id_bbox]
            ids_dict[id_bbox] = value + 1
        else:
            ids_dict[id_bbox] = 1
    #################################################################################

    #mot_imgid = key.replace('.png','')
    #img_name = mot_imgid.split("/")[3]
    #newname = save_path + '/' + img_name + '_mot.png'
    #print(mot_imgid)
    
    #for j in range(track_bbs_ids.shape[0]):  
    #    ele = track_bbs_ids[j, :]
    #    x = int(ele[0])
    #    y = int(ele[1])
    #    x2 = int(ele[2])
    #    y2 = int(ele[3])
    #    track_label = str(int(ele[4])) 
    #    cv2.rectangle(det_img, (x, y), (x2, y2), (0, 255, 255), 4)
    #    cv2.putText(det_img, '#'+track_label, (x+5, y-10), 0,0.6,(0,255,255),thickness=2)
        
    #cv2.imwrite(newname,det_img)

