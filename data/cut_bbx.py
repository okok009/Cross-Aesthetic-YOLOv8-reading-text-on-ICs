import torch
import cv2
import json
import numpy as np
from torchvision.transforms.v2 import Transform


class cut_bbx:
    def __init__(self, data_path, json_path=None, gen_cls='gen'):
        if json_path is not None:
            with open(json_path) as f:
                self.ann = json.load(f)
        self.data_path = data_path
        self.gen_cls = gen_cls
    
    def __call__(self,  reverse:bool=False, img_id:str=None, out_size:tuple=(120, 120), img:torch.Tensor=None, bbx=None, bbox=None):
        '''
        bbx: bounding box image
        bbox: bounding box annotation
        '''
        bbxs = []
        if reverse:
            assert bbox is not None
            bbx = cv2.resize(bbx, (bbox['w'], bbox['h']))
            img = torch.zeros()
            return img
        
        else:
            img = cv2.imread(self.data_path + img_id + '.jpg')
            aesthetic_onehot = torch.zeros([2])
            cls_onehot = torch.zeros([62])
            for i in range(len(self.ann['annotations'])):
                if img_id == str(self.ann['annotations'][i]['image_id']):
                    broken_bbx = {}
                    if self.gen_cls == 'cls':
                        if self.ann['annotations'][i]['aesthetic'][2]:
                            aesthetic_onehot[1] = 1
                        elif self.ann['annotations'][i]['aesthetic'] == [0, 0, 0]:
                            aesthetic_onehot[0] = 1
                        else:
                            continue
                    elif self.gen_cls == 'gen':
                        if self.ann['annotations'][i]['aesthetic'][2] == 0:
                            continue
                    cls_onehot[self.ann['annotations'][i]['category_id']-1] = 1
                    coco_bbox = list(map(int, self.ann['annotations'][i]['bbox']))
                    rotation_degree = -self.ann['annotations'][i]['rotation_degree']
                    center_x, center_y = coco_bbox[0] + coco_bbox[2]/2, coco_bbox[1] + coco_bbox[3]/2
                    M = cv2.getRotationMatrix2D((center_x, center_y), rotation_degree, 1)
                    x1, y1 = np.dot(M, np.array([coco_bbox[0], coco_bbox[1], 1]))
                    x2, y2 = np.dot(M, np.array([coco_bbox[0], coco_bbox[1]+coco_bbox[3], 1]))
                    x3, y3 = np.dot(M, np.array([coco_bbox[0]+coco_bbox[2], coco_bbox[1]+coco_bbox[3], 1]))
                    x4, y4 = np.dot(M, np.array([coco_bbox[0]+coco_bbox[2], coco_bbox[1], 1]))
                    broken_bbx['x1'], broken_bbx['y1'] = int(x1), int(y1)
                    broken_bbx['x2'], broken_bbx['y2'] = int(x2), int(y2)
                    broken_bbx['x3'], broken_bbx['y3'] = int(x3), int(y3)
                    broken_bbx['x4'], broken_bbx['y4'] = int(x4), int(y4)
                    broken_bbx['h'], broken_bbx['w'] = coco_bbox[2], coco_bbox[3]
                    broken_bbx['coco_x'], broken_bbx['coco_y'] = coco_bbox[0], coco_bbox[1]
                    broken_bbx['rotation_degree'] = rotation_degree
                    bbxs.append(broken_bbx)
                    break
            for bbox in bbxs:
                x1, y1, x2, y2 = (bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'])
                x3, y3, x4, y4 = (bbox['x3'], bbox['y3'], bbox['x4'], bbox['y4'])
                pts = np.array([[x1,y1],[x2, y2],[x3, y3],[x4,y4]])
                img_ = np.copy(img)
                cv2.fillPoly(img,[pts], (0, 0, 0))
                filter = np.zeros_like(img)
                filter = filter == img
                filter.dtype = np.int8
                bbx = img_ * filter
                if rotation_degree != 0:
                    M = cv2.getRotationMatrix2D((center_x, center_y), -rotation_degree, 1)
                    bbx = cv2.warpAffine(bbx, M, (bbx.shape[0]*2, bbx.shape[1]*2))
                    bbx = bbx[:img_.shape[0], :img_.shape[1]]
                if bbox['coco_y'] < 0:
                    bbox['coco_y'] = 0
                if bbox['coco_x'] < 0:
                    bbox['coco_x'] = 0
                bbx = bbx[broken_bbx['coco_y']:broken_bbx['coco_y']+bbox['w'], broken_bbx['coco_x']:broken_bbx['coco_x']+bbox['h']]
                if bbx.shape[0] == 0 or bbx.shape[1] == 0:
                    print(img_id)
                bbx = cv2.resize(bbx, out_size)

            return img, bbx, bbox, aesthetic_onehot, cls_onehot

if __name__ == "__main__":
    json_path = 'E:/Datasets/ICText/annotation/GOLD_REF_TRAIN_FINAL.json'
    data_path = 'E:/Datasets/ICText/train2021/'
    cut = cut_bbx(data_path, json_path, 'cls')
    img, bbx, bbox, aesthetic_onehot, cls_onehot = cut(img_id = '382528')
    cv2.imshow('aa', bbx)
    cv2.waitKey(0)
    cv2.destroyAllWindows()