import os
import fitz
import numpy as np
from PIL import Image
from io import BytesIO
from docx import Document
from docx.oxml.ns import qn
from ocr_tables import ocr_image_text_and_tables
from ocr_utils import pad_image

def pdf_to_text_with_inline_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):

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

            image_np = pad_image(
                np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
            )

            ocr_text = ocr_image_text_and_tables(image_np)

            if ocr_text.strip():
                items.append((bbox.y0, bbox.x0, ocr_text.strip()))

        items.sort(key=lambda x: (x[0], x[1]))

        page_text = []
        for _, _, c in items:
            page_text.append(c)

        page_text = "\n\n".join(page_text)

        if page_text.strip():
            pages.append((page_num, page_text))

    return pages

def docx_to_text_with_inline_ocr(docx_path):
    doc = Document(docx_path)

    pages = []
    current_page = []
    page_num = 1

    for para in doc.paragraphs:

        # normal paragraph text
        if para.text.strip():
            current_page.append(para.text)

        # detect page break
        if "w:br" in para._element.xml and 'type="page"' in para._element.xml:
            if current_page:
                pages.append((page_num, "\n".join(current_page)))
            page_num += 1
            current_page = []

        # OCR images inside paragraph
        for run in para.runs:
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

                ocr_text = ocr_image_text_and_tables(image_np)

                if ocr_text.strip():
                    current_page.append(ocr_text)

    if current_page:
        pages.append((page_num, "\n".join(current_page)))

    return pages

def test_ocr(file_path):
    if not os.path.exists(file_path):
        print("❌ File not found")
        exit()
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".png", ".jpg", ".jpeg"]:
        print("\n🖼 OCR on image\n")
        raw_text = ocr_image_text_and_tables(file_path)

    elif ext == ".pdf":
        raw_text = pdf_to_text_with_inline_ocr(file_path)

    elif ext == ".docx":
        raw_text = docx_to_text_with_inline_ocr(file_path)

    else:
        print("❌ Unsupported file type")
        exit()
    return raw_text
if __name__ == "__main__":

    file_path = "ocr_image.jpeg"
    text=test_ocr(file_path)
    print("✅ OCR completed successfully")
    print(text)