import torch
import cv2
from torchvision import transforms
from ultralytics import YOLO

'''
if you want to train on ISW_OBB, please try to do:
1. "return v8OBB_ISWLoss(self, sigma=0.3)" ultralytics/nn/tasks.py line 361
2. "x, backbone_out = self._predict_once(x, profile, visualize, embed)", "return x, backbone_out" ultralytics/nn/tasks.py line 101, 103
3. "if type(m) == SPPF:", "backbone_out = x", "return x,  backbone_out" ultralytics/nn/tasks.py line 125, 126, 135
4. "preds, backbone_output = self.forward(batch["img"]) if preds is None else preds", "return self.criterion(preds, batch, backbone_output)" ultralytics/nn/tasks.py line 265, 267
5. class v8OBB_ISWLoss ultralytics/utils/loss.py line 781~951
6. class v8OBB_CRLoss ultralytics/utils/loss.py line 1052~1267
7. "preds = self.postprocess(preds[0])" ultralytics/engine/validator.py line 186
8. "m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))[0]]) "ultralytics/nn/tasks.py line 297
'''

if __name__ == "__main__":
    print(torch.version)
    device = "cuda:0"

    model = YOLO('E:/ray_workspace/yolov8_master/runs/obb/train_best/weights/best.pt')
    model.to(device)
    model.train(data='E:/ray_workspace/yolov8_master/ictext.yaml', epochs=100, imgsz=1024, batch=3)
    # model.train(data='D:/Datasets/ICText_cls/', epochs=200, imgsz=120, batch=5)
    print('--------------------------')
