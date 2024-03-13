import json

json_path = 'E:/Datasets/ICText/annotation/GOLD_REF_TRAIN_FINAL.json'
with open(json_path) as f:
    ann = json.load(f)

image_id = []
broken_clean_image_id = []

for i in range(len(ann['annotations'])):
    img_name = str(ann['annotations'][i]['image_id'])
    if img_name not in image_id:
        if ann['annotations'][i]['aesthetic'][2] == 1 or ann['annotations'][i]['aesthetic'] == [0, 0, 0]:
            broken_clean_image_id.append(img_name)
        image_id.append(img_name)

broken_image_file = 'broken_clean_img.txt'
with open(broken_image_file, 'a+') as f:
    for i in range(len(broken_clean_image_id)):
        f.write(broken_clean_image_id[i] + '\n')
