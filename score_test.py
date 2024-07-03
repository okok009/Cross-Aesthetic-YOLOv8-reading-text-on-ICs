import numpy as np
import cv2
from utils.score import isw
from ultralytics import YOLO

'''
1. "return results, preds" in ultralytics/models/yolo/detect/predict.py line 44
2. 註解 ultralytics/models/yolo/detect/predict.py line 24~32, 38~42
3. 註解 ultralytics/engine/predictor.py line 293~313
4. "self.results = preds" in ultralytics/engine/predictor.py  line 316
5. "self.results, preds = self.postprocess(preds, im, im0s)" in ultralytics/engine/predictor.py line 291
6. "for i, (f, n, m, args) in enumerate(d["backbone"]):" in ultralytics/nn/task.py line 839
'''

def score_test(original_image = 'D:/Datasets/ICText_hand/772_original.jpg', 
               hand_image = 'D:/Datasets/ICText_hand/772_hand.jpg', 
               original_model = None, 
               model = 'E:/ray_workspace/yolov8_master/yolov8l-obb_without_head.yaml', 
               score = None,
               show = False):
    
    # load image
    original_image = original_image
    hand_image = hand_image

    # model
    model = YOLO(model)
    model_dict = model.model.state_dict()

    # load pretrain weight
    if original_model is not None:
        original_model = YOLO(original_model)
        pretrain_dict = original_model.model.state_dict()
        no_load_key, temp_dict = [], {}
        for k, v in pretrain_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.model.load_state_dict(model_dict)

    # get featuremap
    original_feature = model.predict(original_image, save=False, imgsz=1024, conf=0.5)[0]
    hand_feature = model.predict(hand_image, save=False, imgsz=1024, conf=0.5)[0]

    # score
    loss = score(original_feature, hand_feature, 1)
    print(loss)

    # feature map
    if show:
        original_image = cv2.imread(original_image)
        hand_image = cv2.imread(hand_image)
        for i in range(original_feature.shape[0]):
            
            o_f = original_feature[i].cpu().numpy().astype(np.uint8) * 255
            o_f = cv2.resize(src=o_f, dsize=(1024, 1024))
            o_f = cv2.applyColorMap(o_f, cv2.COLORMAP_JET) + 0.002 * original_image
            o_f = cv2.resize(src=o_f, dsize=(600, 600))
            h_f = hand_feature[i].cpu().numpy().astype(np.uint8) * 255
            h_f = cv2.resize(src=h_f, dsize=(1024, 1024))
            h_f = cv2.applyColorMap(h_f, cv2.COLORMAP_JET) + 0.002 * hand_image
            h_f = cv2.resize(src=h_f, dsize=(600, 600))
            compare = np.concatenate((o_f, h_f), axis=1) 
            cv2.imshow(f'compare-{i}    original                                                                                                                                           |hand', compare)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    original_image = 'D:/Datasets/ICText_hand/777.jpg'
    hand_image = 'D:/Datasets/ICText_hand/777_new.jpg'
    original_model = 'E:/ray_workspace/yolov8_master/runs/obb/train_best/weights/best.pt'
    model = 'E:/ray_workspace/yolov8_master/yolov8l-obb_without_head.yaml'
    score = isw
    show = False
    score_test(original_image, hand_image, original_model, model, score, show)
