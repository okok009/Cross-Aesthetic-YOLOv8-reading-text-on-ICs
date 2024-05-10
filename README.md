# 🌟Cross-Aesthetic-YOLOv8-reading-text-on-ICs
## Introduction

This is a work of oriented object detection on IC Text.

![image](https://github.com/okok009/Cross-Aesthetic-YOLOv8-reading-text-on-ICs/blob/master/assets/Introduction.png)

## Research Flow
In this dataset, we have so many broken or blurry or low contrast data. So I have to analyze this dataset with a model (here I choose YOLOv8), then find a aesthetic that have most error to fix this problem.

![image](https://github.com/okok009/Cross-Aesthetic-YOLOv8-reading-text-on-ICs/blob/master/assets/Research%20Flow.png)

## aesthetic_error_matrix
The “broken” style characters are most worth to deal with, so that Gan model will generate debroken bounding box.

![image](https://github.com/okok009/Cross-Aesthetic-YOLOv8-reading-text-on-ICs/blob/master/assets/aesthetic_error_matrix.png)

## Gan model training
![image](https://github.com/okok009/Cross-Aesthetic-YOLOv8-reading-text-on-ICs/blob/master/assets/Gan%20model%20training.png)

## Result with a simple GAN model
After training, I founded a problem that no matter which input image is the broken style will be the same.

![image](https://github.com/okok009/Cross-Aesthetic-YOLOv8-reading-text-on-ICs/blob/master/assets/Using%20simple%20GAN%20model.png)

## Result with a complex GAN model
So I change the generate model to unt_defnet152. Then the broken style of output will be different with different input image.

![image](https://github.com/okok009/Cross-Aesthetic-YOLOv8-reading-text-on-ICs/blob/master/assets/Using%20complex%20GAN%20model.png)

## Gan model generating
![image](https://github.com/okok009/Cross-Aesthetic-YOLOv8-reading-text-on-ICs/blob/master/assets/Gan%20model%20generating.png)

## Do a normal traing on YOLOv8 with more data
Before retraining YOLOv8 with Covariance Loss, I do a normal training to check if those generated image are useful.

![image](https://github.com/okok009/Cross-Aesthetic-YOLOv8-reading-text-on-ICs/blob/master/assets/new_aesthetic_error_matrix.png)

## Retraining yolov8 backbone
![image](https://github.com/okok009/Cross-Aesthetic-YOLOv8-reading-text-on-ICs/blob/master/assets/Retraining%20yolov8%20backbone.png)

## References
[1] Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics

[2] Ng, Chun Chet, Lin, Che-Tsung, Tan, Zhi Qin, Wang, Xinyu, Kew, Jie Long, Chan, Chee Seng, & Zach, Christopher. (2024). "When IC meets text: Towards a rich annotated integrated circuit text dataset." [Pattern Recognition]. https://doi.org/10.1016/j.patcog.2023.110124

[3] Choi, Sungha, et al. "Robustnet: Improving domain generalization in urban-scene segmentation via instance selective whitening." CVPR. 2021.
