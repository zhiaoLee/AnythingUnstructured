

import os
import shutil

to_path = r"D:\CCKS2025\data\_traindata\traindata"
out_path = r"D:\CCKS2025\output\output_20250702_1"


for name in os.listdir(out_path):
    img_path = os.path.join(out_path, name, "images", "title.jpg")
    level_path = os.path.join(out_path, name, "images", "data.json")
    new_img_path = os.path.join(to_path, "images", f"{name}.jpg")
    new_level_path = os.path.join(to_path, "levels", f"{name}.json")
    os.makedirs(os.path.join(to_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(to_path, "levels"), exist_ok=True)

    shutil.copy(img_path, new_img_path)
    shutil.copy(level_path, new_level_path)

