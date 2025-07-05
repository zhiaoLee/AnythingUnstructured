

import os
import shutil

to_path = r"D:\CCKS2025\data\_traindata\pdf_traindata"
source_path = r"D:\CCKS2025\data\_traindata\pdf_output_data"


for name in os.listdir(source_path):
    img_path = os.path.join(source_path, name, "images", "title.jpg")
    level_path = os.path.join(source_path, name, "images", "data.json")
    new_img_path = os.path.join(to_path, "images", f"{name}.jpg")
    new_level_path = os.path.join(to_path, "levels", f"{name}.json")
    os.makedirs(os.path.join(to_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(to_path, "levels"), exist_ok=True)

    shutil.copy(img_path, new_img_path)
    shutil.copy(level_path, new_level_path)

