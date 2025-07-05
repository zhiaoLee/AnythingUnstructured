import os
import wget
import json
import argparse


def main():

    jsonl_file_path = "build_data/acl_2023_inproceedings.jsonl"
    save_path = "pdfs"
    os.makedirs(save_path, exist_ok=True)
    with open(jsonl_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            ocr_markdown = data['ocr_markdown']
            pdf_url = data["url"]+'.pdf'
            wget.download(pdf_url, out=save_path)


if __name__ == "__main__":
    main()