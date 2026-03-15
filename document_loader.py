import os
import fitz
import numpy as np
from PIL import Image
from io import BytesIO
from pypdf import PdfReader
from docx import Document
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from ocr_text import ocr_image_text_and_tables
from ocr_utils import pad_image

def pdf_to_text_with_inline_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    final_text = []

    for page in doc:
        items = []
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, text, _, block_type = b
            if block_type == 0 and text.strip():
                items.append((y0, x0, text.strip()))

        for img in page.get_images(full=True):
            xref = img[0]
            try:
                bbox = page.get_image_bbox(img)
            except:
                continue

            image_bytes = doc.extract_image(xref)["image"]
            image_np = pad_image(np.array(Image.open(BytesIO(image_bytes)).convert("RGB")))
            ocr_text = ocr_image_text_and_tables(image_np)
            if ocr_text.strip():
                items.append((bbox.y0, bbox.x0, ocr_text.strip()))

        items.sort(key=lambda x: (x[0], x[1]))
        for _, _, c in items:
            final_text.append(c)

    return "\n\n".join(final_text)

def docx_to_text_with_inline_ocr(docx_path):
    doc = Document(docx_path)
    final_text = []

    for para in doc.paragraphs:
        for run in para.runs:
            if run.text:
                final_text.append(run.text)

            for d in run._element.xpath('.//w:drawing'):
                blip = d.xpath('.//a:blip')
                if not blip:
                    continue
                embed = blip[0].get(qn('r:embed'))
                image_part = doc.part.related_parts.get(embed)
                if not image_part:
                    continue
                image_np = pad_image(
                    np.array(Image.open(BytesIO(image_part.blob)).convert("RGB"))
                )
                final_text.append(ocr_image_text_and_tables(image_np))
        final_text.append("\n")

    return "".join(final_text)

def load_document(path):

    if path.lower().endswith(".pdf"):
        return pdf_to_pages(path)

    if path.lower().endswith(".docx"):
        return docx_to_pages(path)

    raise ValueError("Unsupported file")

def docx_to_pages(docx_path):
    doc = Document(docx_path)

    pages = []
    current_page = []
    page_num = 1

    for para in doc.paragraphs:

        # Detect page break
        if "w:br" in para._element.xml and 'type="page"' in para._element.xml:
            if current_page:
                pages.append((page_num, "\n".join(current_page)))
            page_num += 1
            current_page = []
        else:
            if para.text.strip():
                current_page.append(para.text)

    if current_page:
        pages.append((page_num, "\n".join(current_page)))

    return pages

def pdf_to_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append((page_num, text))

    return pages

if __name__ == "__main__":
    print("📄 Loader ready")
