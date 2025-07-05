# Copyright (c) Opendatalab. All rights reserved.
import json
import os.path
import time

from loguru import logger
from src.dict2md.ocr_mkcontent import merge_para_with_text
from openai import OpenAI
import ast
import cv2
import numpy as np
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        r"D:\CCKS2025\weights\model", torch_dtype="auto", device_map="auto"
    )
processor = AutoProcessor.from_pretrained(r"D:\CCKS2025\weights\model")

def get_prompt(num):

    title_optimize_prompt = f"""输入的内容是一篇文档中所有标题组成的图片，共{num}行，请给每行标题确认对应的层级，使结果符合正常文档的层次结构，
    注意：
    1、为每个标题元素添加适当的层次结构
    2、行高较大或字体越浓的标题一般是更高级别的标题
    3、标题从前至后的层级必须是连续的，不能跳过层级
    4、标题层级最多为4级，不要添加过多的层级
    5、优化后的标题只保留代表该标题的层级的整数，不要保留其他信息
    IMPORTANT: 
    请直接返回优化过的由标题层级组成的字典，格式为{{行号:标题层级}}，如下：
    {{0:1,1:2,2:2,3:3}}
    字典的长度必须为{num},不需要对字典格式化，不需要返回任何其他信息。
    Corrected title list:
    """
    return title_optimize_prompt

# 生成白色填充矩阵（上下填充）
def pad_with_white(img, delta_h):
    if delta_h <= 0:
        return img
    # 计算上下填充量（优先填充下方，保持居中）
    top_pad = delta_h // 2
    bottom_pad = delta_h - top_pad
    return cv2.copyMakeBorder(
        img,
        top=top_pad, bottom=bottom_pad, left=10, right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )

def vllm_aided_title(pdf_info_dict, ds, out_path=None):
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     r"D:\CCKS2025\weights\model", torch_dtype="auto", device_map="auto"
    # )
    # processor = AutoProcessor.from_pretrained(r"D:\CCKS2025\weights\model")

    title_dict = {}
    origin_title_list = []
    i = 0
    for page_num, page in pdf_info_dict.items():
        blocks = page["para_blocks"]
        for block in blocks:
            if block["type"] == "title":
                origin_title_list.append(block)
                if page_num in title_dict.keys():
                    title_dict[page_num].append(block)
                else:
                    title_dict[page_num] = [block]

    imgs = []
    #imgs_first_width = []
    max_width = 0
    max_higth = 0
    for page_num, block in title_dict.items():
        page_id = int(page_num.rsplit("_", 1)[-1])
        img_dict = ds.get_page(page_id).get_image()
        real_height = img_dict["height"]
        real_width = img_dict["width"]
        pdf_info = pdf_info_dict[page_num]
        width, height  = pdf_info["page_size"]      #[612.0, 792.0]
        h_radio = real_height / height
        w_radio = real_width / width
        img_arr = img_dict["img"]

        for bk in block:
            bk_img = None
            for i, line in enumerate(bk["lines"]):
                bbox = line["bbox"]
                crop_img = img_arr[int(bbox[1]*h_radio) : int(bbox[3]*h_radio),
                           int(bbox[0]*w_radio) : int(bbox[2]*w_radio)]

                if i == 0:
                    bk_img = crop_img
                else:
                    h1, w1, c1 = bk_img.shape
                    h2, w2, c2 = crop_img.shape
                    max_height = max(h1, h2)
                    delta_h1 = max_height - h1
                    delta_h2 = max_height - h2

                    img1_padded = pad_with_white(bk_img, delta_h1)
                    img2_padded = pad_with_white(crop_img, delta_h2)
                    bk_img = np.hstack((img1_padded, img2_padded))

            imgs.append(bk_img)
            #imgs_first_width.append(int(bk["bbox"][0]*w_radio))
            h, w, _ = bk_img.shape
            max_width = max_width if max_width > w else w
            max_higth += h + 30

    all_img = np.full((max_higth+200, max_width+200, 3), (255, 255, 255), dtype=np.uint8)

    cur_height = 40
    for index in range(len(imgs)):
        img = imgs[index]
        #first_w = imgs_first_width[index]
        h1, w1, c1 = img.shape
        all_img[cur_height:cur_height + h1, 50: 50+w1] = img
        cur_height += h1 + 30

    import re
    import base64
    cv2.imwrite(os.path.join(out_path, "title.jpg"), all_img)
    with open(os.path.join(out_path, "title.jpg"), 'rb') as f:
        img_base64 = base64.b64encode(f.read()).decode()

    # import base64
    # img_byte_arr = cv2.imencode('.png', all_img)[1].tobytes()
    #
    # # Encode the image to Base64
    # img_base64 = base64.b64encode(img_byte_arr)

    # logger.info(f"Title list: {title_dict}")

    title_optimize_prompt = get_prompt(len(imgs))

    retry_count = 0
    max_retries = 3
    dict_completion = None

    while retry_count < max_retries:
        try:
            # completion = client.chat.completions.create(
            #     model="Qwen/Qwen2.5-VL-32B-Instruct",
            #     messages=[
            #         {'role': 'user',
            #          'content': [
            #              {"type": "text", "text": title_optimize_prompt},
            #              {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
            #          ]}],
            #     temperature=0.1,
            # )
            messages = [
                {'role': 'user',
                 'content': [
                     {"type": "text", "text": title_optimize_prompt},
                     {"type": "image", "image_url": f"{os.path.join(out_path, "title.jpg")}"},
                 ]}]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            logger.info(f"Title completion: {output_text[0]}")
            completion_content = output_text[0]
            clean_dict_str = completion_content.strip('`\n').replace('python\n', '').replace('json\n', '').replace('，',',').replace('：', ':')

            try:
                dict_completion = ast.literal_eval(clean_dict_str)
            except Exception as e:
                logger.error(f"ast.literal_eval Error: {e}")
                pattern = r'(\{.*?\})'  # 匹配完整的 {} 结构，包括嵌套内容
                match = re.search(pattern, clean_dict_str, re.DOTALL)  # re.DOTALL 允许匹配换行符
                if match:
                    json_str = match.group(1)
                    try:
                        try:
                            dict_completion = ast.literal_eval(json_str)
                        except:
                            key_value_pairs = json_str.strip('{}').split(',')
                            # 2. 初始化字典
                            dict_completion = {}

                            # 3. 解析每个键值对
                            for pair in key_value_pairs:
                                # 去除键值对中的空格
                                pair = pair.strip()
                                # 分割键和值（按第一个冒号分割）
                                key_str, value_str = pair.split(':', 1)
                                # 转换为整数
                                key = int(key_str)
                                value = int(value_str)
                                # 添加到字典
                                dict_completion[key] = value
                                dict_completion[key] = value
                    except json.JSONDecodeError:
                        print("非合法 JSON 格式")
            print("提取的字典:", dict_completion)
            print("类型:", type(dict_completion))
            import json

            with open(os.path.join(out_path, "data.json"), "w", encoding="utf-8") as f:
                json.dump(dict_completion, f, ensure_ascii=False, indent=4)

            print("字典已成功保存为JSON文件！")
            if dict_completion and len(dict_completion) == len(origin_title_list):
                for i, origin_title_block in enumerate(origin_title_list):
                    try:
                        origin_title_block["level"] = int(dict_completion[str(i)])
                    except Exception:
                        origin_title_block["level"] = int(dict_completion[i])
                break
            else:
                logger.warning("The number of titles in the optimized result is not equal to the number of titles in the input.")
                retry_count += 1
        except Exception as e:
            logger.exception(e)
            retry_count += 1

    if dict_completion is None:
        logger.error("Failed to decode dict after maximum retries.")

if __name__ == '__main__':
    import json
    from src.data.data_reader_writer import FileBasedDataReader
    from src.data.dataset import PymuDocDataset

    pdf_file_name = r"D:\CCKS2025\code\AnythingUnstructured\demo\pdfs\772e98a9-b55d-4ae3-ae05-b2b3de84cf5d.pdf"
    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(pdf_file_name)  # read the pdf content
    # proc
    ## Create Dataset Instance
    ds = PymuDocDataset(pdf_bytes)

    # 加载 JSON 文件
    with open(os.path.join(pdf_file_name.rsplit(".")[0].replace("pdfs","output"), "pdf_info_dict.json"), "r", encoding="utf-8") as f:
        pdf_info_dict = json.load(f)
    vllm_aided_title(pdf_info_dict, ds, os.path.join(pdf_file_name.rsplit(".")[0].replace("pdfs","output")))