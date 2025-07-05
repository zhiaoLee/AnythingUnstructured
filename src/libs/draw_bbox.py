import random

import cv2
import fitz
from src.config.constants import CROSS_PAGE
from src.config.ocr_content_type import (BlockType, CategoryId,
                                         ContentType)
from src.data.dataset import Dataset
from src.model.magic_model import MagicModel


def draw_bbox_without_number(i, bbox_list, page, rgb_config, fill_config):
    new_rgb = []
    for item in rgb_config:
        item = float(item) / 255
        new_rgb.append(item)
    page_data = bbox_list[i]
    for bbox in page_data:
        x0, y0, x1, y1 = bbox
        rect_coords = fitz.Rect(x0, y0, x1, y1)  # Define the rectangle
        if fill_config:
            page.draw_rect(
                rect_coords,
                color=None,
                fill=new_rgb,
                fill_opacity=0.3,
                width=0.5,
                overlay=True,
            )  # Draw the rectangle
        else:
            page.draw_rect(
                rect_coords,
                color=new_rgb,
                fill=None,
                fill_opacity=1,
                width=0.5,
                overlay=True,
            )  # Draw the rectangle


def draw_bbox_with_number(i, bbox_list, page, rgb_config, fill_config, draw_bbox=True):
    new_rgb = []
    for item in rgb_config:
        item = float(item) / 255
        new_rgb.append(item)
    page_data = bbox_list[i]
    for j, bbox in enumerate(page_data):
        x0, y0, x1, y1 = bbox
        rect_coords = fitz.Rect(x0, y0, x1, y1)  # Define the rectangle
        if draw_bbox:
            if fill_config:
                page.draw_rect(
                    rect_coords,
                    color=None,
                    fill=new_rgb,
                    fill_opacity=0.3,
                    width=0.5,
                    overlay=True,
                )  # Draw the rectangle
            else:
                page.draw_rect(
                    rect_coords,
                    color=new_rgb,
                    fill=None,
                    fill_opacity=1,
                    width=0.5,
                    overlay=True,
                )  # Draw the rectangle
        page.insert_text(
            (x1 + 2, y0 + 10), str(j + 1), fontsize=10, color=new_rgb
        )  # Insert the index in the top left corner of the rectangle


def draw_layout_bbox(pdf_info, pdf_bytes, out_path, filename):
    dropped_bbox_list = []
    tables_list, tables_body_list = [], []
    tables_caption_list, tables_footnote_list = [], []
    imgs_list, imgs_body_list, imgs_caption_list = [], [], []
    imgs_footnote_list = []
    titles_list = []
    texts_list = []
    interequations_list = []
    lists_list = []
    indexs_list = []
    for page in pdf_info:

        page_dropped_list = []
        tables, tables_body, tables_caption, tables_footnote = [], [], [], []
        imgs, imgs_body, imgs_caption, imgs_footnote = [], [], [], []
        titles = []
        texts = []
        interequations = []
        lists = []
        indices = []

        for dropped_bbox in page['discarded_blocks']:
            page_dropped_list.append(dropped_bbox['bbox'])
        dropped_bbox_list.append(page_dropped_list)
        for block in page['para_blocks']:
            bbox = block['bbox']
            if block['type'] == BlockType.Table:
                tables.append(bbox)
                for nested_block in block['blocks']:
                    bbox = nested_block['bbox']
                    if nested_block['type'] == BlockType.TableBody:
                        tables_body.append(bbox)
                    elif nested_block['type'] == BlockType.TableCaption:
                        tables_caption.append(bbox)
                    elif nested_block['type'] == BlockType.TableFootnote:
                        tables_footnote.append(bbox)
            elif block['type'] == BlockType.Image:
                imgs.append(bbox)
                for nested_block in block['blocks']:
                    bbox = nested_block['bbox']
                    if nested_block['type'] == BlockType.ImageBody:
                        imgs_body.append(bbox)
                    elif nested_block['type'] == BlockType.ImageCaption:
                        imgs_caption.append(bbox)
                    elif nested_block['type'] == BlockType.ImageFootnote:
                        imgs_footnote.append(bbox)
            elif block['type'] == BlockType.Title:
                titles.append(bbox)
            elif block['type'] == BlockType.Text:
                texts.append(bbox)
            elif block['type'] == BlockType.InterlineEquation:
                interequations.append(bbox)
            elif block['type'] == BlockType.List:
                lists.append(bbox)
            elif block['type'] == BlockType.Index:
                indices.append(bbox)

        tables_list.append(tables)
        tables_body_list.append(tables_body)
        tables_caption_list.append(tables_caption)
        tables_footnote_list.append(tables_footnote)
        imgs_list.append(imgs)
        imgs_body_list.append(imgs_body)
        imgs_caption_list.append(imgs_caption)
        imgs_footnote_list.append(imgs_footnote)
        titles_list.append(titles)
        texts_list.append(texts)
        interequations_list.append(interequations)
        lists_list.append(lists)
        indexs_list.append(indices)

    layout_bbox_list = []

    table_type_order = {
        'table_caption': 1,
        'table_body': 2,
        'table_footnote': 3
    }
    for page in pdf_info:
        page_block_list = []
        for block in page['para_blocks']:
            if block['type'] in [
                BlockType.Text,
                BlockType.Title,
                BlockType.InterlineEquation,
                BlockType.List,
                BlockType.Index,
            ]:
                bbox = block['bbox']
                page_block_list.append(bbox)
            elif block['type'] in [BlockType.Image]:
                for sub_block in block['blocks']:
                    bbox = sub_block['bbox']
                    page_block_list.append(bbox)
            elif block['type'] in [BlockType.Table]:
                sorted_blocks = sorted(block['blocks'], key=lambda x: table_type_order[x['type']])
                for sub_block in sorted_blocks:
                    bbox = sub_block['bbox']
                    page_block_list.append(bbox)

        layout_bbox_list.append(page_block_list)

    pdf_docs = fitz.open('pdf', pdf_bytes)

    for i, page in enumerate(pdf_docs):

        draw_bbox_without_number(i, dropped_bbox_list, page, [158, 158, 158], True)
        # draw_bbox_without_number(i, tables_list, page, [153, 153, 0], True)  # color !
        draw_bbox_without_number(i, tables_body_list, page, [204, 204, 0], True)
        draw_bbox_without_number(i, tables_caption_list, page, [255, 255, 102], True)
        draw_bbox_without_number(i, tables_footnote_list, page, [229, 255, 204], True)
        # draw_bbox_without_number(i, imgs_list, page, [51, 102, 0], True)
        draw_bbox_without_number(i, imgs_body_list, page, [153, 255, 51], True)
        draw_bbox_without_number(i, imgs_caption_list, page, [102, 178, 255], True)
        draw_bbox_without_number(i, imgs_footnote_list, page, [255, 178, 102], True),
        draw_bbox_without_number(i, titles_list, page, [102, 102, 255], True)
        draw_bbox_without_number(i, texts_list, page, [153, 0, 76], True)
        draw_bbox_without_number(i, interequations_list, page, [0, 255, 0], True)
        draw_bbox_without_number(i, lists_list, page, [40, 169, 92], True)
        draw_bbox_without_number(i, indexs_list, page, [40, 169, 92], True)

        draw_bbox_with_number(
            i, layout_bbox_list, page, [255, 0, 0], False, draw_bbox=False
        )

    # Save the PDF
    pdf_docs.save(f'{out_path}/{filename}')


def draw_span_bbox(pdf_info, pdf_bytes, out_path, filename):
    text_list = []
    inline_equation_list = []
    interline_equation_list = []
    image_list = []
    table_list = []
    dropped_list = []
    next_page_text_list = []
    next_page_inline_equation_list = []

    def get_span_info(span):
        if span['type'] == ContentType.Text:
            if span.get(CROSS_PAGE, False):
                next_page_text_list.append(span['bbox'])
            else:
                page_text_list.append(span['bbox'])
        elif span['type'] == ContentType.InlineEquation:
            if span.get(CROSS_PAGE, False):
                next_page_inline_equation_list.append(span['bbox'])
            else:
                page_inline_equation_list.append(span['bbox'])
        elif span['type'] == ContentType.InterlineEquation:
            page_interline_equation_list.append(span['bbox'])
        elif span['type'] == ContentType.Image:
            page_image_list.append(span['bbox'])
        elif span['type'] == ContentType.Table:
            page_table_list.append(span['bbox'])

    for page in pdf_info:
        page_text_list = []
        page_inline_equation_list = []
        page_interline_equation_list = []
        page_image_list = []
        page_table_list = []
        page_dropped_list = []

        # 将跨页的span放到移动到下一页的列表中
        if len(next_page_text_list) > 0:
            page_text_list.extend(next_page_text_list)
            next_page_text_list.clear()
        if len(next_page_inline_equation_list) > 0:
            page_inline_equation_list.extend(next_page_inline_equation_list)
            next_page_inline_equation_list.clear()

        # 构造dropped_list
        for block in page['discarded_blocks']:
            if block['type'] == BlockType.Discarded:
                for line in block['lines']:
                    for span in line['spans']:
                        page_dropped_list.append(span['bbox'])
        dropped_list.append(page_dropped_list)
        # 构造其余useful_list
        # for block in page['para_blocks']:  # span直接用分段合并前的结果就可以
        for block in page['preproc_blocks']:
            if block['type'] in [
                BlockType.Text,
                BlockType.Title,
                BlockType.InterlineEquation,
                BlockType.List,
                BlockType.Index,
            ]:
                for line in block['lines']:
                    for span in line['spans']:
                        get_span_info(span)
            elif block['type'] in [BlockType.Image, BlockType.Table]:
                for sub_block in block['blocks']:
                    for line in sub_block['lines']:
                        for span in line['spans']:
                            get_span_info(span)
        text_list.append(page_text_list)
        inline_equation_list.append(page_inline_equation_list)
        interline_equation_list.append(page_interline_equation_list)
        image_list.append(page_image_list)
        table_list.append(page_table_list)
    pdf_docs = fitz.open('pdf', pdf_bytes)
    for i, page in enumerate(pdf_docs):
        # 获取当前页面的数据
        draw_bbox_without_number(i, text_list, page, [255, 0, 0], False)
        draw_bbox_without_number(i, inline_equation_list, page, [0, 255, 0], False)
        draw_bbox_without_number(i, interline_equation_list, page, [0, 0, 255], False)
        draw_bbox_without_number(i, image_list, page, [255, 204, 0], False)
        draw_bbox_without_number(i, table_list, page, [204, 0, 255], False)
        draw_bbox_without_number(i, dropped_list, page, [158, 158, 158], False)

    # Save the PDF
    pdf_docs.save(f'{out_path}/{filename}')


def draw_model_bbox(model_list, dataset: Dataset, out_path, filename):
    dropped_bbox_list = []
    tables_body_list, tables_caption_list, tables_footnote_list = [], [], []
    imgs_body_list, imgs_caption_list, imgs_footnote_list = [], [], []
    titles_list = []
    texts_list = []
    interequations_list = []
    magic_model = MagicModel(model_list, dataset)
    for i in range(len(model_list)):
        page_dropped_list = []
        tables_body, tables_caption, tables_footnote = [], [], []
        imgs_body, imgs_caption, imgs_footnote = [], [], []
        titles = []
        texts = []
        interequations = []
        page_info = magic_model.get_model_list(i)
        layout_dets = page_info['layout_dets']
        for layout_det in layout_dets:
            bbox = layout_det['bbox']
            if layout_det['category_id'] == CategoryId.Text:
                texts.append(bbox)
            elif layout_det['category_id'] == CategoryId.Title:
                titles.append(bbox)
            elif layout_det['category_id'] == CategoryId.TableBody:
                tables_body.append(bbox)
            elif layout_det['category_id'] == CategoryId.TableCaption:
                tables_caption.append(bbox)
            elif layout_det['category_id'] == CategoryId.TableFootnote:
                tables_footnote.append(bbox)
            elif layout_det['category_id'] == CategoryId.ImageBody:
                imgs_body.append(bbox)
            elif layout_det['category_id'] == CategoryId.ImageCaption:
                imgs_caption.append(bbox)
            elif layout_det['category_id'] == CategoryId.InterlineEquation_YOLO:
                interequations.append(bbox)
            elif layout_det['category_id'] == CategoryId.Abandon:
                page_dropped_list.append(bbox)
            elif layout_det['category_id'] == CategoryId.ImageFootnote:
                imgs_footnote.append(bbox)

        tables_body_list.append(tables_body)
        tables_caption_list.append(tables_caption)
        tables_footnote_list.append(tables_footnote)
        imgs_body_list.append(imgs_body)
        imgs_caption_list.append(imgs_caption)
        titles_list.append(titles)
        texts_list.append(texts)
        interequations_list.append(interequations)
        dropped_bbox_list.append(page_dropped_list)
        imgs_footnote_list.append(imgs_footnote)

    for i in range(len(dataset)):
        page = dataset.get_page(i)
        draw_bbox_with_number(
            i, dropped_bbox_list, page, [158, 158, 158], True
        )  # color !
        draw_bbox_with_number(i, tables_body_list, page, [204, 204, 0], True)
        draw_bbox_with_number(i, tables_caption_list, page, [255, 255, 102], True)
        draw_bbox_with_number(i, tables_footnote_list, page, [229, 255, 204], True)
        draw_bbox_with_number(i, imgs_body_list, page, [153, 255, 51], True)
        draw_bbox_with_number(i, imgs_caption_list, page, [102, 178, 255], True)
        draw_bbox_with_number(i, imgs_footnote_list, page, [255, 178, 102], True)
        draw_bbox_with_number(i, titles_list, page, [102, 102, 255], True)
        draw_bbox_with_number(i, texts_list, page, [153, 0, 76], True)
        draw_bbox_with_number(i, interequations_list, page, [0, 255, 0], True)

    # Save the PDF
    dataset.dump_to_file(f'{out_path}/{filename}')


def draw_line_sort_bbox(pdf_info, pdf_bytes, out_path, filename):
    layout_bbox_list = []

    for page in pdf_info:
        page_line_list = []
        for block in page['preproc_blocks']:
            if block['type'] in [BlockType.Text]:
                for line in block['lines']:
                    bbox = line['bbox']
                    index = line['index']
                    page_line_list.append({'index': index, 'bbox': bbox})
            elif block['type'] in [BlockType.Title, BlockType.InterlineEquation]:
                if 'virtual_lines' in block:
                    if len(block['virtual_lines']) > 0 and block['virtual_lines'][0].get('index', None) is not None:
                        for line in block['virtual_lines']:
                            bbox = line['bbox']
                            index = line['index']
                            page_line_list.append({'index': index, 'bbox': bbox})
                else:
                    for line in block['lines']:
                        bbox = line['bbox']
                        index = line['index']
                        page_line_list.append({'index': index, 'bbox': bbox})
            elif block['type'] in [BlockType.Image, BlockType.Table]:
                for sub_block in block['blocks']:
                    if sub_block['type'] in [BlockType.ImageBody, BlockType.TableBody]:
                        if len(sub_block['virtual_lines']) > 0 and sub_block['virtual_lines'][0].get('index', None) is not None:
                            for line in sub_block['virtual_lines']:
                                bbox = line['bbox']
                                index = line['index']
                                page_line_list.append({'index': index, 'bbox': bbox})
                        else:
                            for line in sub_block['lines']:
                                bbox = line['bbox']
                                index = line['index']
                                page_line_list.append({'index': index, 'bbox': bbox})
                    elif sub_block['type'] in [BlockType.ImageCaption, BlockType.TableCaption, BlockType.ImageFootnote, BlockType.TableFootnote]:
                        for line in sub_block['lines']:
                            bbox = line['bbox']
                            index = line['index']
                            page_line_list.append({'index': index, 'bbox': bbox})
        sorted_bboxes = sorted(page_line_list, key=lambda x: x['index'])
        layout_bbox_list.append(sorted_bbox['bbox'] for sorted_bbox in sorted_bboxes)
    pdf_docs = fitz.open('pdf', pdf_bytes)
    for i, page in enumerate(pdf_docs):
        draw_bbox_with_number(i, layout_bbox_list, page, [255, 0, 0], False)

    pdf_docs.save(f'{out_path}/{filename}')


def draw_char_bbox(pdf_bytes, out_path, filename):
    pdf_docs = fitz.open('pdf', pdf_bytes)
    for i, page in enumerate(pdf_docs):
        for block in page.get_text('rawdict', flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP)['blocks']:
            for line in block['lines']:
                for span in line['spans']:
                    for char in span['chars']:
                        char_bbox = char['bbox']
                        page.draw_rect(char_bbox, color=[1, 0, 0], fill=None, fill_opacity=1, width=0.3, overlay=True,)
    pdf_docs.save(f'{out_path}/{filename}')


def draw_detection_boxes(img, detections, output_path=None, color=(0, 255, 0), thickness=2, font_scale=0.7):
    """
    在图片上绘制检测框和类别标签
    参数：
    img: 图片
    detections: 检测结果列表，每个元素为字典，包含：
        - 'bbox': 边界框坐标 [x1, y1, x2, y2]（左上角和右下角坐标）
        - 'class': 类别名称
    output_path: 输出图片路径，若为None则直接显示图片
    color: 框的颜色（BGR格式）
    thickness: 框的线条粗细
    font_scale: 字体大小缩放因子
    """
    # 读取图片
    # 遍历每个检测结果
    newimg = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = str(det['class'])
        random.seed(det['class'])
        color = (random.choices(range(1,255))[0],random.choices(range(1,255))[0],random.choices(range(1,255))[0])

        # 绘制矩形框
        cv2.rectangle(newimg, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        # 绘制类别标签（左上角）
        (text_width, text_height), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                              thickness)
        cv2.rectangle(newimg, (int(x1), int(y1) - text_height - baseline),
                      (int(x1) + text_width, int(y1)), color, -1)  # 填充矩形背景
        cv2.putText(newimg, class_name, (int(x1), int(y1) - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness=3)  # 黑色文本

    # 保存或显示图片
    cv2.imwrite(output_path, newimg)
