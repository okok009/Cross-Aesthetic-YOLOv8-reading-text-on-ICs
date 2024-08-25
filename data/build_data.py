import cv2
import os
import json
import numpy as np
import torch
import torch.nn as nn
from cut_bbx import cut_bbx
from torchvision.io import read_image
from torchvision.transforms import v2


def ann_json(json_path, data_path, classes_id, rotate_img, rotate_bbx, img_name=None, ictext2doa=False):
    '''
    當資料集的annotation 是存成與coco相同格式的.json時,可以使用下面方式。
    '''
    with open(json_path) as f:
        ann = json.load(f)
    with open(classes_id) as f:
        classes = f.read().splitlines()

    image_id = []

    for i in range(len(ann['annotations'])):
        if img_name is None:
            img_name = str(ann['annotations'][i]['image_id'])

        if img_name not in image_id:
            image_id.append(img_name)
            bbxs = []
            img = cv2.imread(data_path + img_name + '.jpg')

            if rotate_img:
                '''
                根據當前(第i個)bbx的旋轉角度,並非真實圖像的旋轉角度,因為一張圖會有多種角度的文字。
                '''
                rotation_degree = ann['annotations'][i]['rotation_degree']
                center_y = img.shape[0]/2
                center_x = img.shape[1]/2
                M = cv2.getRotationMatrix2D((center_x, center_y), rotation_degree, 1)
                hw = int(center_x*2) if center_x > center_y else int(center_y*2)
                img = cv2.warpAffine(img, M, (hw, hw))

            for j in range(len(ann['annotations'])):

                if img_name == str(ann['annotations'][j]['image_id']):
                    bbox = list(map(int, ann['annotations'][j]['bbox']))
                    category_id = ann['annotations'][j]['category_id']
                    bbx = {}
                    bbx['label'] = classes[category_id-1]

                    if rotate_bbx:
                        rotation_degree = -ann['annotations'][j]['rotation_degree']
                        center_x, center_y = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2
                        M = cv2.getRotationMatrix2D((center_x, center_y), rotation_degree, 1)
                        x1, y1 = np.dot(M, np.array([bbox[0], bbox[1], 1]))
                        x2, y2 = np.dot(M, np.array([bbox[0], bbox[1]+bbox[3], 1]))
                        x3, y3 = np.dot(M, np.array([bbox[0]+bbox[2], bbox[1]+bbox[3], 1]))
                        x4, y4 = np.dot(M, np.array([bbox[0]+bbox[2], bbox[1], 1]))
                        bbx['x1'], bbx['y1'] = int(x1), int(y1)
                        bbx['x2'], bbx['y2'] = int(x2), int(y2)
                        bbx['x3'], bbx['y3'] = int(x3), int(y3)
                        bbx['x4'], bbx['y4'] = int(x4), int(y4)

                    else:
                        bbx['x1'] = bbox[0]
                        bbx['y1'] = bbox[1]
                        bbx['x2'] = bbx['x1'] + bbox[2]
                        bbx['y2'] = bbx['y1'] + bbox[3]
                    
                    bbxs.append(bbx)

            for bbox in bbxs:
                label = bbox['label']

                if rotate_bbx:
                    x1, y1, x2, y2 = (bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'])
                    x3, y3, x4, y4 = (bbox['x3'], bbox['y3'], bbox['x4'], bbox['y4'])
                    pts = np.array([[x1,y1],[x2, y2],[x3, y3],[x4,y4]])
                    cv2.polylines(img,[pts],True,(0,255,0),2)
                else:
                    x1, y1, x2, y2 = (bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'])
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

                fontFace = cv2.FONT_HERSHEY_COMPLEX
                fontScale = 0.5
                thickness = 1
                labelSize = cv2.getTextSize(label, fontFace, fontScale, thickness)
                _x1 = x1 # bottomleft x of text
                _y1 = y1 # bottomleft y of text
                _x2 = x1+labelSize[0][0] # topright x of text
                _y2 = y1-labelSize[0][1] # topright y of text
                cv2.rectangle(img, (_x1,_y1), (_x2,_y2), (0,255,0), cv2.FILLED) # text background
                cv2.putText(img, label, (x1,y1), fontFace, fontScale, (0,0,0), thickness)
                if ictext2doa:
                    label_file = f'E:/Datasets/ICText/labels/{img_name}.txt'
                    with open(label_file, 'a+') as f:
                        f.write(str(bbox['label']) + ' ' + str(bbox['x1']) + ' ' + str(bbox['y1']) + ' ' + str(bbox['x2']) + ' ' + str(bbox['y2']) + ' ' + str(bbox['x3']) + ' ' + str(bbox['y3']) + ' ' + str(bbox['x4']) + ' ' + str(bbox['y4']) + '\n')
            
            cv2.imshow(img_name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        img_name = None


def ictext2doa(json_path, data_path, img_name=None):
    with open(json_path) as f:
        ann = json.load(f)

    image_id = []

    for i in range(len(ann['annotations'])):
        if img_name is None:
            img_name = str(ann['annotations'][i]['image_id'])

        if img_name not in image_id:
            image_id.append(img_name)
            bbxs = []
            image = read_image(data_path + img_name + '.jpg')
            if image.shape[-1] != image.shape[-2]:
                pad_size = abs(image.shape[-1]-image.shape[-2])
                pad = nn.ZeroPad2d((0, pad_size, 0, 0)) if image.shape[-1] < image.shape[-2] else nn.ZeroPad2d((0, 0, 0, pad_size))
                image = pad(image)

            for j in range(len(ann['annotations'])):

                if img_name == str(ann['annotations'][j]['image_id']):
                    category_id = ann['annotations'][j]['category_id']
                    bbx = {}
                    bbox = torch.tensor(list(map(int, ann['annotations'][j]['bbox'])))
                    bbx['label'] = str(ann['annotations'][j]['category_id'] - 1)
                    rotation_degree = -ann['annotations'][j]['rotation_degree']
                    bbox[2:] += bbox[:2]
                    bbx['bbox'] = bbox
                    bbxs.append(bbx)
            if bbxs == []:
                continue    
            
            labels = []
            bboxs = bbxs[0]['bbox'].unsqueeze(0)
            labels.append(bbxs[0]['label'])
            for j in range(1, len(bbxs)):
                if bbxs[j] is None:
                    break
                bboxs = torch.cat((bboxs, bbxs[j]['bbox'].unsqueeze(0)), 0)
                labels.append(bbxs[j]['label'])

            transform = v2.Compose([
                # v2.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                v2.Resize((1024, 1024), antialias=True)]
            )

            bboxs = bboxs * (1024/image.shape[-1])
            image = transform(image)
            image = image.permute(1, 2, 0)
            img = image.numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite('E:/Datasets/ICText/new_data/'+img_name + '.jpg', img)
            
            for i, bbox in enumerate(bboxs):
                center_x, center_y = int(bbox[0] + (bbox[2]-bbox[0])/2), int(bbox[1] + (bbox[3]-bbox[1])/2)
                M = cv2.getRotationMatrix2D((center_x, center_y), rotation_degree, 1)
                x1, y1 = np.dot(M, np.array([int(bbox[0]), int(bbox[1]), 1]))
                x2, y2 = np.dot(M, np.array([int(bbox[0]), int(bbox[3]), 1]))
                x3, y3 = np.dot(M, np.array([int(bbox[2]), int(bbox[3]), 1]))
                x4, y4 = np.dot(M, np.array([int(bbox[2]), int(bbox[1]), 1]))

                label_file = f'E:/Datasets/ICText/labels/{img_name}.txt'
                with open(label_file, 'a+') as f:
                    f.write(labels[i] + ' ' + str(format(x1/1024, '.6f')) + ' ' + str(format(y1/1024, '.6f')) + ' ' + str(format(x2/1024, '.6f')) + ' ' + str(format(y2/1024, '.6f')) + ' ' + str(format(x3/1024, '.6f')) + ' ' + str(format(y3/1024, '.6f')) + ' ' + str(format(x4/1024, '.6f')) + ' ' + str(format(y4/1024, '.6f')) + '\n')
            
                pts = np.array([[x1,y1],[x2, y2],[x3, y3],[x4,y4]], dtype=np.int32)
                cv2.polylines(img,[pts],True,(0,255,0),2)
        img_name = None
            # cv2.imshow('aa', img)
            # cv2.waitKey(0)
                
def ictext2cls(img_txt, json_path, data_path, img_name=None):
    # with open('E:/ray_workspace/CrossAestheticYOLOv8/data/' + img_txt + '.txt') as f:
    #     img_ids = f.readlines()
    img_ids = os.listdir(data_path)
    cut = cut_bbx(data_path, json_path, 'cls')
    for img_id in img_ids:
        cut(img_id = img_id[:-4], img_txt=img_txt)

if __name__ == '__main__':

    # imgs = os.listdir('E:/Datasets/ExDark/img')
    # img_path = 'E:/Datasets/ExDark/img/'
    # ann_path = 'E:/Datasets/ExDark/ExDark_Annno/'
    # ann_txt(img_path, ann_path, imgs)
    '''
    img_txt: 格式為xxx_oo..._img, 其中xxx可以是tra或val表示是訓練集或測試集, oo...則可以是clean, broken, only_broken, not_broken, not_only_broken以上四種, 以便區分現在要建立何種資料.
    '''
    img_txt = 'val_not_only_broken_img' # 必須要是tra_開頭表示train，不是打錯!!!
    json_path = 'D:/Datasets/ICText/annotation/GOLD_REF_VAL_FINAL.json'
    data_path = 'D:/Datasets/ICText/val2021/'
    classes_id = 'D:/Datasets/ICText/annotation/classes.txt'
    rotate_img = False
    rotate_bbx = True
    # ann_json(json_path, data_path, classes_id, rotate_img, rotate_bbx)
    ictext2cls(img_txt, json_path, data_path)