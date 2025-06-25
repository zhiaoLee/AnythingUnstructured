import os
import cv2
from src.data.dataset import Dataset
from src.dict2md.ocr_mkcontent import merge_para_with_text

def aided_title(pdf_info_dict, dataset: Dataset, imgpath):

    os.makedirs(imgpath, exist_ok=True)
    title_dict = {}
    origin_title_list = []
    line_avg_heights = []
    title_texts = []
    levels = []
    images_dict = {}
    i = 0
    for page_num, page in pdf_info_dict.items():
        blocks = page["para_blocks"]
        for block in blocks:
            if block["type"] == "title":
                if "level" in block.keys():
                    levels.append(block["level"])
                origin_title_list.append(block)
                title_text = merge_para_with_text(block)
                title_texts.append(title_text)
                page_line_height_list = []
                for line in block['lines']:
                    bbox = line['bbox']
                    page_line_height_list.append(int(bbox[3] - bbox[1]))
                if len(page_line_height_list) > 0:
                    line_avg_height = sum(page_line_height_list) / len(page_line_height_list)
                else:
                    line_avg_height = int(block['bbox'][3] - block['bbox'][1])
                line_avg_heights.append(line_avg_height)
                # title_dict[f"{i}"] = [title_text, line_avg_height, int(page_num[5:]) + 1]
                i += 1
        index = int(page_num.split("_")[-1])
        if index not in images_dict.keys():
            page_data = dataset.get_page(index)
            img_dict = page_data.get_image()
            images_dict[page_num] = img_dict['img']

    last_index = 0
    for origin_title_block in origin_title_list:
        lines = origin_title_block["lines"]
        page_num = origin_title_block["page_num"]
        #level = origin_title_block["level"]
        bbox = origin_title_block["bbox"]

        index = int(page_num.split("_")[-1])
        if index != last_index:
            if index != 0:
                cv2.imwrite(os.path.join(imgpath,page_num+'.jpg'), img)
            img = images_dict[page_num]
        img = cv2.rectangle(
            img,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            color=(255, 0, 0),  # 蓝色（BGR格式）
            thickness=2  # 线宽2像素
        )


if __name__ == '__main__':
    import json
    from src.data.data_reader_writer import FileBasedDataReader
    from src.data.dataset import PymuDocDataset

    pdf_file_name = r"D:\CCKS2025\code\AnythingUnstructured\demo\pdfs\0ef9db04-86f2-4319-b08e-dcb5385b1232.pdf"
    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(pdf_file_name)  # read the pdf content
    # proc
    ## Create Dataset Instance
    ds = PymuDocDataset(pdf_bytes)

    # 加载 JSON 文件
    with open(r"D:\CCKS2025\code\AnythingUnstructured\demo\output\0ef9db04-86f2-4319-b08e-dcb5385b1232\pdf_info_dict.json", "r", encoding="utf-8") as f:
        pdf_info_dict = json.load(f)
    aided_title(pdf_info_dict, ds, r"D:\CCKS2025\code\AnythingUnstructured\demo\output\0ef9db04-86f2-4319-b08e-dcb5385b1232\images\title")