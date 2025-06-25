import os.path
from loguru import logger
import cv2
from paddleocr import LayoutDetection
from tqdm import tqdm
from src.model.sub_modules.layout.pp_doclayout.label_type import PP_DocLayout_plus_L

class PPDocLayoutModel(object):
    def __init__(self, weight, device):
        self.model_name = os.path.basename(weight)
        self.model = LayoutDetection(model_dir=weight)
        self.device = device

    def predict(self, image):
        logger.info("layout: PPDocLayoutModel")
        layout_res = []
        doclayout_yolo_res = self.model.predict(
            image,
            imgsz=1280,
            conf=0.10,
            iou=0.45,
            verbose=False, device=self.device
        )[0]
        for xyxy, conf, cla in zip(
            doclayout_yolo_res.boxes.xyxy.cpu(),
            doclayout_yolo_res.boxes.conf.cpu(),
            doclayout_yolo_res.boxes.cls.cpu(),
        ):
            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
            new_item = {
                "category_id": int(cla.item()),
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                "score": round(float(conf.item()), 3),
            }
            layout_res.append(new_item)
        return layout_res

    def batch_predict(self, images: list, batch_size: int) -> list:
        logger.info("layout: PPDocLayoutModel")
        images_layout_res = []
        # for index in range(0, len(images), batch_size):
        for index in tqdm(range(0, len(images), batch_size), desc="Layout Predict"):
            doclayout_yolo_res = self.model.predict(images[index : index + batch_size], batch_size=batch_size, layout_nms=True)
            for image_res in doclayout_yolo_res:
                image_res_boxes = image_res["boxes"]
                layout_res = []
                for res_box in image_res_boxes:

                    label = res_box["label"]
                    if self.model_name == "PP-DocLayout_plus-L":
                        category_id = PP_DocLayout_plus_L[label]

                    coordinate = res_box["coordinate"]
                    xmin, ymin, xmax,ymax = coordinate
                    score = round(float(res_box["score"]), 3)
                    new_item = {
                        "category_id": category_id,
                        "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                        "score": score,
                    }
                    layout_res.append(new_item)
                images_layout_res.append(layout_res)

        return images_layout_res

if __name__ == "__main__":
    import fitz
    from src.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
    from src.data.dataset import Doc
    from src.data.utils import fitz_doc_to_image
    from src.libs.draw_bbox import draw_detection_boxes

    weight = r"D:\CCKS2025\code\AnythingUnstructured\weights\paddleocrv3\layout\PP-DocLayout_plus-L"
    device = "cuda"
    layout = PPDocLayoutModel(weight=weight, device=device)



    pdf_file_name = r"D:\CCKS2025\code\AnythingUnstructured\demo\pdfs\0ef9db04-86f2-4319-b08e-dcb5385b1232.pdf"
    basename = os.path.basename(pdf_file_name).rsplit(".",-1)[0]
    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(pdf_file_name)  # read the pdf content
    raw_fitz = fitz.open('pdf', pdf_bytes)
    records = [Doc(v) for v in raw_fitz]

    images = []
    for index, doc in enumerate(records):
        img = fitz_doc_to_image(doc._doc)
        images.append(img["img"])
    images_layout_res = layout.batch_predict(images, batch_size=1)
    #draw_layout_bbox(pdf_info, pdf_bytes, out_path, filename):
    for index, (img, layout_res) in enumerate(zip(images, images_layout_res)):
        detections = []
        for lay in layout_res:
            poly = lay["poly"]
            detections.append({
                'bbox': [poly[0], poly[1], poly[2], poly[-1]],
                'class': lay["category_id"]
            })
        output_path = os.path.join("D:\CCKS2025\code\AnythingUnstructured\output\layout", basename)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        draw_detection_boxes(img, detections, output_path=os.path.join(output_path, f"{index}.jpg"), color=(0, 255, 0), thickness=2, font_scale=0.7)