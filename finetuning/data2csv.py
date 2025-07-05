
import os
import json
import pandas as pd

dataset_path = r"D:\CCKS2025\data\_traindata\traindata"
csv_path = r'D:\CCKS2025\data\_traindata\title_train.csv'

images_path = os.path.join(dataset_path, "images")
levels_path = os.path.join(dataset_path, "levels")

image_paths = []
texts = []
for img in os.listdir(images_path):
    img_path = os.path.join(images_path, img)
    level_path = os.path.join(levels_path, img.rsplit(".",1)[0] + ".json")


    with open(level_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_paths.append(img_path)
    texts.append(data)


df = pd.DataFrame({
        'image_path': image_paths,
        'text': texts,
    })


# 将数据保存为CSV文件
df.to_csv(csv_path, index=False)
