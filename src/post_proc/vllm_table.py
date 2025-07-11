from openai import OpenAI
import base64

client = OpenAI(api_key="123", base_url="http://124.88.174.116:32010/v1")

model = "/root/epfs/model/Nanonets-OCR-s"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def ocr_page_with_nanonets_s(img_base64):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                    {
                        "type": "text",
                        "text": "Extract the text from the above document as if you were reading it naturally. "
                                "Return the tables in html format. Return the equations in LaTeX representation. "
                                "If there is an image in the document and image caption is not present, "
                                "add a small description of the image inside the <img></img> tag; "
                                "otherwise, add the image caption inside <img></img>. "
                                "Watermarks should be wrapped in brackets. "
                                "Ex: <watermark>OFFICIAL COPY</watermark>. "
                                "Page numbers should be wrapped in brackets. "
                                "Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. "
                                "Prefer using ☐ and ☑ for check boxes.",
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=15000
    )
    return response.choices[0].message.content

# test_img_path = r"D:\CCKS2025\code\AnythingUnstructured\demo\1751555838.812922.jpg"
# img_base64 = encode_image(test_img_path)
# print(ocr_page_with_nanonets_s(img_base64))
