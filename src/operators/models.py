import copy
import json
import os
from typing import Callable

from src.config.constants import PARSE_TYPE_OCR, PARSE_TYPE_TXT
from src.config.enums import SupportedPdfParseMethod
from src.data.data_reader_writer import DataWriter
from src.data.dataset import Dataset
from src.libs.draw_bbox import draw_model_bbox
from src.libs.version import __version__
from src.operators.pipes import PipeResult
from src.pdf_parse_union_core_v2 import pdf_parse_union
from src.operators import InferenceResultBase

class InferenceResult(InferenceResultBase):
    def __init__(self, inference_results: list, dataset: Dataset):
        """Initialized method.

        Args:
            inference_results (list): the inference result generated by model
            dataset (Dataset): the dataset related with model inference result
        """
        self._infer_res = inference_results
        self._dataset = dataset

    def draw_model(self, file_path: str) -> None:
        """Draw model inference result.

        Args:
            file_path (str): the output file path
        """
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        draw_model_bbox(
            copy.deepcopy(self._infer_res), self._dataset, dir_name, base_name
        )

    def dump_model(self, writer: DataWriter, file_path: str):
        """Dump model inference result to file.

        Args:
            writer (DataWriter): writer handle
            file_path (str): the location of target file
        """
        writer.write_string(
            file_path, json.dumps(self._infer_res, ensure_ascii=False, indent=4)
        )

    def get_infer_res(self):
        """Get the inference result.

        Returns:
            list: the inference result generated by model
        """
        return self._infer_res

    def apply(self, proc: Callable, *args, **kwargs):
        """Apply callable method which.

        Args:
            proc (Callable): invoke proc as follows:
                proc(inference_result, *args, **kwargs)

        Returns:
            Any: return the result generated by proc
        """
        return proc(copy.deepcopy(self._infer_res), *args, **kwargs)

    def pipe_txt_mode(
        self,
        imageWriter: DataWriter,
        start_page_id=0,
        end_page_id=None,
        debug_mode=False,
        lang=None,
    ) -> PipeResult:
        """Post-proc the model inference result, Extract the text using the
        third library, such as `pymupdf`

        Args:
            imageWriter (DataWriter): the image writer handle
            start_page_id (int, optional): Defaults to 0. Let user select some pages He/She want to process
            end_page_id (int, optional):  Defaults to the last page index of dataset. Let user select some pages He/She want to process
            debug_mode (bool, optional): Defaults to False. will dump more log if enabled
            lang (str, optional): Defaults to None.

        Returns:
            PipeResult: the result
        """

        def proc(*args, **kwargs) -> PipeResult:
            res = pdf_parse_union(*args, **kwargs)
            res['_parse_type'] = PARSE_TYPE_TXT
            res['_version_name'] = __version__
            if 'lang' in kwargs and kwargs['lang'] is not None:
                res['lang'] = kwargs['lang']
            return PipeResult(res, self._dataset)

        res = self.apply(
            proc,
            self._dataset,
            imageWriter,
            SupportedPdfParseMethod.TXT,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            debug_mode=debug_mode,
            lang=lang,
        )
        return res

    def pipe_ocr_mode(
        self,
        imageWriter: DataWriter,
        start_page_id=0,
        end_page_id=None,
        debug_mode=False,
        lang=None,
    ) -> PipeResult:
        """Post-proc the model inference result, Extract the text using `OCR`
        technical.

        Args:
            imageWriter (DataWriter): the image writer handle
            start_page_id (int, optional): Defaults to 0. Let user select some pages He/She want to process
            end_page_id (int, optional):  Defaults to the last page index of dataset. Let user select some pages He/She want to process
            debug_mode (bool, optional): Defaults to False. will dump more log if enabled
            lang (str, optional): Defaults to None.

        Returns:
            PipeResult: the result
        """

        def proc(*args, **kwargs) -> PipeResult:
            res = pdf_parse_union(*args, **kwargs)
            res['_parse_type'] = PARSE_TYPE_OCR
            res['_version_name'] = __version__
            if 'lang' in kwargs and kwargs['lang'] is not None:
                res['lang'] = kwargs['lang']
            return PipeResult(res, self._dataset)

        res = self.apply(
            proc,
            self._dataset,
            imageWriter,
            SupportedPdfParseMethod.OCR,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            debug_mode=debug_mode,
            lang=lang,
        )
        return res
