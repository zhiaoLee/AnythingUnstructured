import os.path

import cv2
from doclayout_yolo import YOLOv10
from tqdm import tqdm


class DocLayoutYOLOModel(object):
    def __init__(self, weight, device):
        self.model = YOLOv10(weight)
        self.device = device

    def predict(self, image):
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
        images_layout_res = []
        # for index in range(0, len(images), batch_size):
        for index in tqdm(range(0, len(images), batch_size), desc="Layout Predict"):
            doclayout_yolo_res = [
                image_res.cpu()
                for image_res in self.model.predict(
                    images[index : index + batch_size],
                    imgsz=1280,
                    conf=0.10,
                    iou=0.45,
                    verbose=False,
                    device=self.device,
                )
            ]
            for image_res in doclayout_yolo_res:
                layout_res = []
                for xyxy, conf, cla in zip(
                    image_res.boxes.xyxy,
                    image_res.boxes.conf,
                    image_res.boxes.cls,
                ):
                    xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                    new_item = {
                        "category_id": int(cla.item()),
                        "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                        "score": round(float(conf.item()), 3),
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

    weight = r"weights\\PDF-Extract-Kit-1___0/models\\Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt"
    device = "cuda"
    layout = DocLayoutYOLOModel(weight=weight, device=device)

    pdf_file_name = r"demo\pdfs\0c626c7f-9d1e-4137-87ca-29453764f654.pdf"
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
        output_path = os.path.join(r"D:\CCKS2025\code\MinerU\Out\layout", f"lay_out_{index}.jpg")
        draw_detection_boxes(img, detections, output_path=output_path, color=(0, 255, 0), thickness=2, font_scale=0.7)