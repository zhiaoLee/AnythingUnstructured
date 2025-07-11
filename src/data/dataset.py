import os
from abc import ABC, abstractmethod
from typing import Callable, Iterator

import fitz
from loguru import logger

from src.config.enums import SupportedPdfParseMethod
from src.data.schemas import PageInfo
from src.data.utils import fitz_doc_to_image
from src.filter import classify


class PageableData(ABC):
    @abstractmethod
    def get_image(self) -> dict:
        """Transform data to image."""
        pass

    @abstractmethod
    def get_doc(self) -> fitz.Page:
        """Get the pymudoc page."""
        pass

    @abstractmethod
    def get_page_info(self) -> PageInfo:
        """Get the page info of the page.

        Returns:
            PageInfo: the page info of this page
        """
        pass

    @abstractmethod
    def draw_rect(self, rect_coords, color, fill, fill_opacity, width, overlay):
        """draw rectangle.

        Args:
            rect_coords (list[float]): four elements array contain the top-left and bottom-right coordinates, [x0, y0, x1, y1]
            color (list[float] | None): three element tuple which describe the RGB of the board line, None means no board line
            fill (list[float] | None): fill the board with RGB, None means will not fill with color
            fill_opacity (float): opacity of the fill, range from [0, 1]
            width (float): the width of board
            overlay (bool): fill the color in foreground or background. True means fill in background.
        """
        pass

    @abstractmethod
    def insert_text(self, coord, content, fontsize, color):
        """insert text.

        Args:
            coord (list[float]): four elements array contain the top-left and bottom-right coordinates, [x0, y0, x1, y1]
            content (str): the text content
            fontsize (int): font size of the text
            color (list[float] | None):  three element tuple which describe the RGB of the board line, None will use the default font color!
        """
        pass


class Dataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """The length of the dataset."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[PageableData]:
        """Yield the page data."""
        pass

    @abstractmethod
    def supported_methods(self) -> list[SupportedPdfParseMethod]:
        """The methods that this dataset support.

        Returns:
            list[SupportedPdfParseMethod]: The supported methods, Valid methods are: OCR, TXT
        """
        pass

    @abstractmethod
    def data_bits(self) -> bytes:
        """The bits used to create this dataset."""
        pass

    @abstractmethod
    def get_page(self, page_id: int) -> PageableData:
        """Get the page indexed by page_id.

        Args:
            page_id (int): the index of the page

        Returns:
            PageableData: the page doc object
        """
        pass

    @abstractmethod
    def dump_to_file(self, file_path: str):
        """Dump the file.

        Args:
            file_path (str): the file path
        """
        pass

    @abstractmethod
    def apply(self, proc: Callable, *args, **kwargs):
        """Apply callable method which.

        Args:
            proc (Callable): invoke proc as follows:
                proc(self, *args, **kwargs)

        Returns:
            Any: return the result generated by proc
        """
        pass

    @abstractmethod
    def classify(self) -> SupportedPdfParseMethod:
        """classify the dataset.

        Returns:
            SupportedPdfParseMethod: _description_
        """
        pass

    @abstractmethod
    def clone(self):
        """clone this dataset."""
        pass


class PymuDocDataset(Dataset):
    def __init__(self, bits: bytes, lang=None):
        """Initialize the dataset, which wraps the pymudoc documents.

        Args:
            bits (bytes): the bytes of the pdf
        """
        self._raw_fitz = fitz.open('pdf', bits)
        self._records = [Doc(v) for v in self._raw_fitz]
        self._data_bits = bits
        self._raw_data = bits
        self._classify_result = None

        if lang == '':
            self._lang = None
        elif lang == 'auto':
            from src.model.sub_modules.language_detection.utils import \
                auto_detect_lang
            self._lang = auto_detect_lang(self._data_bits)
            logger.info(f'lang: {lang}, detect_lang: {self._lang}')
        else:
            self._lang = lang
            logger.info(f'lang: {lang}')

    def __len__(self) -> int:
        """The page number of the pdf."""
        return len(self._records)

    def __iter__(self) -> Iterator[PageableData]:
        """Yield the page doc object."""
        return iter(self._records)

    def supported_methods(self) -> list[SupportedPdfParseMethod]:
        """The method supported by this dataset.

        Returns:
            list[SupportedPdfParseMethod]: the supported methods
        """
        return [SupportedPdfParseMethod.OCR, SupportedPdfParseMethod.TXT]

    def data_bits(self) -> bytes:
        """The pdf bits used to create this dataset."""
        return self._data_bits

    def get_page(self, page_id: int) -> PageableData:
        """The page doc object.

        Args:
            page_id (int): the page doc index

        Returns:
            PageableData: the page doc object
        """
        return self._records[page_id]

    def dump_to_file(self, file_path: str):
        """Dump the file.

        Args:
            file_path (str): the file path
        """

        dir_name = os.path.dirname(file_path)
        if dir_name not in ('', '.', '..'):
            os.makedirs(dir_name, exist_ok=True)
        self._raw_fitz.save(file_path)

    def apply(self, proc: Callable, *args, **kwargs):
        """Apply callable method which.

        Args:
            proc (Callable): invoke proc as follows:
                proc(dataset, *args, **kwargs)

        Returns:
            Any: return the result generated by proc
        """
        if 'lang' in kwargs and self._lang is not None:
            kwargs['lang'] = self._lang
        return proc(self, *args, **kwargs)

    def classify(self) -> SupportedPdfParseMethod:
        """classify the dataset.

        Returns:
            SupportedPdfParseMethod: _description_
        """
        if self._classify_result is None:
            self._classify_result = classify(self._data_bits)
        return self._classify_result

    def clone(self):
        """clone this dataset."""
        return PymuDocDataset(self._raw_data)

    def set_images(self, images):
        for i in range(len(self._records)):
            self._records[i].set_image(images[i])

class ImageDataset(Dataset):
    def __init__(self, bits: bytes, lang=None):
        """Initialize the dataset, which wraps the pymudoc documents.

        Args:
            bits (bytes): the bytes of the photo which will be converted to pdf first. then converted to pymudoc.
        """
        pdf_bytes = fitz.open(stream=bits).convert_to_pdf()
        self._raw_fitz = fitz.open('pdf', pdf_bytes)
        self._records = [Doc(v) for v in self._raw_fitz]
        self._raw_data = bits
        self._data_bits = pdf_bytes

        if lang == '':
            self._lang = None
        elif lang == 'auto':
            from src.model.sub_modules.language_detection.utils import \
                auto_detect_lang
            self._lang = auto_detect_lang(self._data_bits)
            logger.info(f'lang: {lang}, detect_lang: {self._lang}')
        else:
            self._lang = lang
            logger.info(f'lang: {lang}')

    def __len__(self) -> int:
        """The length of the dataset."""
        return len(self._records)

    def __iter__(self) -> Iterator[PageableData]:
        """Yield the page object."""
        return iter(self._records)

    def supported_methods(self):
        """The method supported by this dataset.

        Returns:
            list[SupportedPdfParseMethod]: the supported methods
        """
        return [SupportedPdfParseMethod.OCR]

    def data_bits(self) -> bytes:
        """The pdf bits used to create this dataset."""
        return self._data_bits

    def get_page(self, page_id: int) -> PageableData:
        """The page doc object.

        Args:
            page_id (int): the page doc index

        Returns:
            PageableData: the page doc object
        """
        return self._records[page_id]

    def dump_to_file(self, file_path: str):
        """Dump the file.

        Args:
            file_path (str): the file path
        """
        dir_name = os.path.dirname(file_path)
        if dir_name not in ('', '.', '..'):
            os.makedirs(dir_name, exist_ok=True)
        self._raw_fitz.save(file_path)

    def apply(self, proc: Callable, *args, **kwargs):
        """Apply callable method which.

        Args:
            proc (Callable): invoke proc as follows:
                proc(dataset, *args, **kwargs)

        Returns:
            Any: return the result generated by proc
        """
        return proc(self, *args, **kwargs)

    def classify(self) -> SupportedPdfParseMethod:
        """classify the dataset.

        Returns:
            SupportedPdfParseMethod: _description_
        """
        return SupportedPdfParseMethod.OCR

    def clone(self):
        """clone this dataset."""
        return ImageDataset(self._raw_data)

    def set_images(self, images):
        for i in range(len(self._records)):
            self._records[i].set_image(images[i])

class Doc(PageableData):
    """Initialized with pymudoc object."""

    def __init__(self, doc: fitz.Page):
        self._doc = doc
        self._img = None

    def get_image(self):
        """Return the image info.

        Returns:
            dict: {
                img: np.ndarray,
                width: int,
                height: int
            }
        """
        if self._img is None:
            self._img = fitz_doc_to_image(self._doc)
        return self._img

    def set_image(self, img):
        """
        Args:
            img (np.ndarray): the image
        """
        if self._img is None:
            self._img = img

    def get_doc(self) -> fitz.Page:
        """Get the pymudoc object.

        Returns:
            fitz.Page: the pymudoc object
        """
        return self._doc

    def get_page_info(self) -> PageInfo:
        """Get the page info of the page.

        Returns:
            PageInfo: the page info of this page
        """
        page_w = self._doc.rect.width
        page_h = self._doc.rect.height
        return PageInfo(w=page_w, h=page_h)

    def __getattr__(self, name):
        if hasattr(self._doc, name):
            return getattr(self._doc, name)

    def draw_rect(self, rect_coords, color, fill, fill_opacity, width, overlay):
        """draw rectangle.

        Args:
            rect_coords (list[float]): four elements array contain the top-left and bottom-right coordinates, [x0, y0, x1, y1]
            color (list[float] | None): three element tuple which describe the RGB of the board line, None means no board line
            fill (list[float] | None): fill the board with RGB, None means will not fill with color
            fill_opacity (float): opacity of the fill, range from [0, 1]
            width (float): the width of board
            overlay (bool): fill the color in foreground or background. True means fill in background.
        """
        self._doc.draw_rect(
            rect_coords,
            color=color,
            fill=fill,
            fill_opacity=fill_opacity,
            width=width,
            overlay=overlay,
        )

    def insert_text(self, coord, content, fontsize, color):
        """insert text.

        Args:
            coord (list[float]): four elements array contain the top-left and bottom-right coordinates, [x0, y0, x1, y1]
            content (str): the text content
            fontsize (int): font size of the text
            color (list[float] | None):  three element tuple which describe the RGB of the board line, None will use the default font color!
        """
        self._doc.insert_text(coord, content, fontsize=fontsize, color=color)