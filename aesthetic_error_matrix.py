import cv2
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.transforms import v2
from ultralytics import YOLO

'''
If you want to discriminate aesthetic_error_matrix, 
you should do three things:
1. "return results, preds" in ultralytics/models/yolo/obb/predict.py line 54
2. "self.results = preds" in ultralytics/engine/predictor.py  line 316
3. "self.results, preds = self.postprocess(preds, im, im0s)" in ultralytics/engine/predictor.py line 291
4. check task.py
'''

def rectangle_intersection(rect1, rect2):
    '''
    rect1: ((x, y), (w, h), r) 
    rect2: ((x, y), (w, h), r) 
    return rect_inter
    '''
    inter_points = cv2.rotatedRectangleIntersection(rect1, rect2)
    if inter_points[1] is None:
        inter_area = 0
    else:
        inter_area = cv2.contourArea(inter_points[1])
    
    return inter_area

def aesthetic_error_matrix(model, train_val, img_name=None):
    lc_cl_num = 0  # number of class error with low contrast
    lc_un_num = 0  # number of unpredicted with low contrast
    bl_cl_num = 0  # number of class error with blurring
    bl_un_num = 0  # number of unpredicted with blurring
    br_cl_num = 0  # number of class error with broken
    br_un_num = 0  # number of unpredicted with broken

    if train_val == 'train':
        with open('D:/Datasets/ICText/annotation/GOLD_REF_TRAIN_FINAL.json') as f:
            ann = json.load(f)
    elif train_val == 'val':
        with open('D:/Datasets/ICText/annotation/GOLD_REF_VAL_FINAL.json') as f:
            ann = json.load(f)
    else:
        raise ValueError(f'"should give "train" or "val", but got {train_val}')

    image_id = []
    for i in range(len(ann['annotations'])):
        # if img_name is None:
        img_name = str(ann['annotations'][i]['image_id'])
        if img_name not in image_id:
            image_id.append(img_name)
            bbxs = []
            for j in range(len(ann['annotations'])):

                if img_name == str(ann['annotations'][j]['image_id']):
                    coco_bbox       = list(map(int, ann['annotations'][j]['bbox']))
                    category_id     = ann['annotations'][j]['category_id']
                    rotation_degree = ann['annotations'][j]['rotation_degree'] * (-1)
                    aesthetic       = ann['annotations'][j]['aesthetic']
                    center_x, center_y = coco_bbox[0] + coco_bbox[2]/2, coco_bbox[1] + coco_bbox[3]/2
                    for k in range(len(ann['images'])):
                        if img_name == str(ann['images'][k]['id']):
                            image_shape = ann['images'][k]['height'] if ann['images'][k]['height'] > ann['images'][k]['width'] else ann['images'][k]['width']
                            image_ratio = (1024/image_shape)
                    bbx = {}
                    bbx['label'] = category_id-1
                    bbx['x'], bbx['y'] = image_ratio*int(center_x), image_ratio*int(center_y)
                    bbx['w'], bbx['h'] = image_ratio*int(coco_bbox[2]), image_ratio*int(coco_bbox[3])
                    bbx['r'] = rotation_degree
                    bbx['aesthetic'] = aesthetic
                    bbxs.append(bbx)
            
            # model predict
            preds = model.predict(f'D:/Datasets/ICText_yolov8/images/{train_val}/{img_name}.jpg', save=False, imgsz=1024, conf=0.5)[0]
            print(preds.shape)
            preds = preds.detach().cpu().numpy()
            for bbx in bbxs:
                if bbx['aesthetic'][0]:
                    lc_un_num += 1
                if bbx['aesthetic'][1]:
                    bl_un_num += 1
                if bbx['aesthetic'][2]:
                    br_un_num += 1
                if bbx['aesthetic'] != [0, 0, 0]:
                    rect1 = ((bbx['x'], bbx['y']), (bbx['w'], bbx['h']), bbx['r'])
                    rect1_area = bbx['w'] * bbx['h']
                    for i in range(len(preds)):
                        rect2 = ((int(preds[i][0]), int(preds[i][1])), (int(preds[i][2]), int(preds[i][3])), preds[i][6]* (180/np.pi))
                        rect2_area = preds[i][2] * preds[i][3]
                        inter_area = rectangle_intersection(rect1, rect2)
                        iou = inter_area / (rect1_area + rect2_area - inter_area)
                        if iou < 0.5:
                            continue

                        # class error and unpredted error
                        if bbx['aesthetic'][0]:
                            if bbx['label'] != preds[i][5]:
                                lc_cl_num   += 1
                            lc_un_num += -1

                        if bbx['aesthetic'][1]:
                            if bbx['label'] != preds[i][5]:
                                bl_cl_num   += 1
                            bl_un_num += -1

                        if bbx['aesthetic'][2]:
                            if bbx['label'] != preds[i][5]:
                                br_cl_num   += 1
                            br_un_num += -1
    error_matrix = np.array([[lc_cl_num, bl_cl_num, br_cl_num], 
                             [lc_un_num, bl_un_num, br_un_num]])
    return error_matrix


if __name__ == '__main__':
    # backbone = YOLO('runs/obb/train_with_generate_data/weights/best.pt')
    backbone = None
    
    model = YOLO('runs/obb/train_with_CLoss_2/weights/best.pt')
    model_dict = model.model.state_dict()

    # load pretrain weight
    if backbone is not None:
        backbone_dict = backbone.model.state_dict()
        no_load_key, temp_dict = [], {}
        for k, v in backbone_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.model.load_state_dict(model_dict)

    train_val = 'train'
    train_error_matrix = aesthetic_error_matrix(model, train_val)
    train_val = 'val'
    val_error_matrix = aesthetic_error_matrix(model, train_val)
    print(f'train:{train_error_matrix}')
    print(f'val:{val_error_matrix}')
