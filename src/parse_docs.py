import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from unstructured.partition.pdf import partition_pdf
import os
from tqdm import tqdm

def parse_pdf(file_path):
    """
    Extrae texto y tablas de un PDF.
    Si el texto es escaso, aplica OCR a las páginas (por ejemplo, circuitos escaneados).
    """
    elements = partition_pdf(filename=file_path)
    text_content = "\n".join([el.text for el in elements if hasattr(el, "text") and el.text])
    
    if len(text_content.strip()) < 100:
        print(f"[OCR] Aplicando OCR a {file_path}")
        images = convert_from_path(file_path)
        text_ocr = []
        for img in tqdm(images, desc="OCR pages"):
            text_ocr.append(pytesseract.image_to_string(img))
        text_content = "\n".join(text_ocr)
    
    return text_content
