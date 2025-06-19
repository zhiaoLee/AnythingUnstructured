# Copyright (c) Opendatalab. All rights reserved.
import os

from src.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from src.data.dataset import PymuDocDataset
from src.model.doc_analyze_by_custom_model import doc_analyze
from src.config.enums import SupportedPdfParseMethod


def pdf2md(pdf_file_name, name_without_extension):
    # args
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    # pdf_file_name = os.path.join(__dir__, "pdfs", "demo1.pdf")  # replace with the real pdf path
    # name_without_extension = os.path.basename(pdf_file_name).split('.')[0]

    # prepare env
    # 设置输出的图片和Markdown文档的路径
    local_image_dir = os.path.join(__dir__, "output_0616_1", name_without_extension, "images")
    local_md_dir = os.path.join(__dir__, "output_0616_1", name_without_extension)
    image_dir = str(os.path.basename(local_image_dir))
    os.makedirs(local_image_dir, exist_ok=True)

    # 定义图片和Markdown文档写出工具类
    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

    # read bytes
    # 定义读入PDF文档的工具类
    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(pdf_file_name)  # read the pdf content

    # proc
    ## Create Dataset Instance
    #创建提取PDF的Dataset
    ds = PymuDocDataset(pdf_bytes)

    ## inference
    ## 判断PDF是否是扫描版
    if ds.classify() == SupportedPdfParseMethod.OCR:
        # 扫描版PDF使用OCR提取，doc_analyze方法是实现提取的关键方法，包括加载模型和实现提取
        infer_result = ds.apply(doc_analyze, ocr=True)

        ## pipeline
        pipe_result = infer_result.pipe_ocr_mode(image_writer)

    else:
        # 普通版PDF提取方式
        infer_result = ds.apply(doc_analyze, ocr=False)

        ## pipeline
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    ### get model inference result
    ### 获取模型推理结果
    model_inference_result = infer_result.get_infer_res()

    ### draw layout result on each page
    ### 在每页上绘制Layout模型处理的结果
    pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_extension}_layout.pdf"))

    ### draw spans result on each page
    ### 绘制每页的跨距结果
    pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_extension}_spans.pdf"))

    ### get markdown content
    ### 获取Markdown的内
    md_content = pipe_result.get_markdown(image_dir)

    ### dump markdown
    pipe_result.dump_md(md_writer, f"{name_without_extension}.md", image_dir)

    return md_content


if __name__ == "__main__":
    import pandas as pd

    data = []
    path = r"D:\CCKS2025\data\dataset_A"
    for i, name in enumerate(os.listdir(path)):
        print(f"第{i+1}张, 名字是{name}......")
        file_id = name.rsplit(".", 1)[0]
        pdf_path = os.path.join(path, name)
        md_content = pdf2md(pdf_path, file_id)
        data.append([file_id, md_content])
    df = pd.DataFrame(data, columns=["file_id", "answer"])
    # 保存为 CSV 文件，文件名为 'example.csv'
    df.to_csv('result/output_20250616_1.csv')
    print(df)
