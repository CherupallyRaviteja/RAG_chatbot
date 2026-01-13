# ================= GENERALIZED OCR PRINT PROGRAM =================

import os
import re
import cv2
import tempfile
from pypdf import PdfReader
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

import os
import warnings

# -------- SILENCE ALL LOGS --------
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # TensorFlow (if indirectly used)
os.environ["FLAGS_log_level"] = "3"          # Paddle internal logs
os.environ["GLOG_minloglevel"] = "3"         # GLOG (used by Paddle)
os.environ["PYTHONWARNINGS"] = "ignore"      # Python warnings
os.environ["KMP_WARNINGS"] = "off"           # MKL warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""      # Disable GPU logs (CPU only)

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
POPPLER_PATH = None   # set path if poppler is not in PATH (Windows)
LANG = "en"

# ---------------- OCR INIT ----------------
ocr = PaddleOCR(
    lang=LANG,
    use_angle_cls=True   # ✅ valid in all recent versions
)

# ---------------- PREPROCESS ----------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # PaddleOCR expects 3-channel image
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# ---------------- HELPERS ----------------
def is_scanned_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    for page in reader.pages[:2]:
        if page.extract_text():
            return False
    return True
def rows_to_markdown(rows):
    """
    Converts detected table rows into Markdown table.
    Assumes first row is header.
    """
    if not rows:
        return ""

    header = rows[0]
    sep = ["---"] * len(header)

    md = []
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(sep) + " |")

    for row in rows[1:]:
        md.append("| " + " | ".join(row) + " |")

    return "\n".join(md)



def ocr_image_text_and_tables(image_path):
    """
    Robust OCR:
    - Handles new & old PaddleOCR outputs
    - Processes text + table-like rows
    - RETURNS extracted text (string)
    """

    result = ocr.ocr(image_path)
    if not result:
        return ""

    items = []

    # ---- New PaddleOCR (dict-based) ----
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        for page in result:
            polys = page.get("dt_polys", [])
            texts = page.get("rec_texts", [])

            for poly, text in zip(polys, texts):
                x = min(p[0] for p in poly)
                y = min(p[1] for p in poly)
                items.append((y, x, text))

    # ---- Old PaddleOCR (list-based) ----
    else:
        for line in result:
            box = line[0]
            text = line[1][0]

            x = min(p[0] for p in box)
            y = min(p[1] for p in box)
            items.append((y, x, text))

    items.sort()
    lines_out = []
    table_rows = []
    current_row = []
    last_y = None
    ROW_GAP = 18

    for y, x, text in items:
        if last_y is None or abs(y - last_y) <= ROW_GAP:
            current_row.append(text)
        else:
            lines_out.append(_format_row(current_row))
            current_row = [text]
        last_y = y

    if current_row:
        if len(current_row) >= 3:
            table_rows.append(current_row)
        else:
            lines_out.append(" ".join(current_row))

    # convert detected table rows to markdown
    if table_rows:
        lines_out.append("\n" + rows_to_markdown(table_rows) + "\n")

    return "\n".join(lines_out)



def _format_row(row):
    if len(row) >= 3:
        return " | ".join(row)   # table-like
    else:
        return " ".join(row)     # normal text

def format_row_rowwise(row):
    """
    Keeps columns together, moves to next line per row.
    """
    return " ".join(row)

def _print_row(row):
    """
    Decides whether a row is table-like or plain text
    """
    if len(row) >= 3:
        # Likely a table row
        print(" | ".join(row))
    else:
        # Normal text line
        print(" ".join(row))

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def html_to_markdown(html):
    html = re.sub(r"\n", "", html)

    rows = re.findall(r"<tr>(.*?)</tr>", html)
    table = []



    for row in rows:
        cells = re.findall(r"<t[dh]>(.*?)</t[dh]>", row)
        cells = [re.sub(r"<.*?>", "", c).strip() for c in cells]
        table.append(cells)

    if not table:
        return ""

    header = table[0]
    sep = ["---"] * len(header)

    md = []
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(sep) + " |")

    for row in table[1:]:
        md.append("| " + " | ".join(row) + " |")

    return "\n".join(md)

def ocr_image(image_path):
    processed = preprocess_image(image_path)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, processed)
        tmp_path = tmp.name

    result = ocr.predict(tmp_path)
    os.remove(tmp_path)

    lines = []

    # ---- CASE 1: New PaddleOCR Pipeline ----
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        for page in result:
            texts = page.get("rec_texts", [])
            lines.extend(texts)

    # ---- CASE 2: Old PaddleOCR ----
    else:
        for line in result:
            if isinstance(line, list) and len(line) >= 2:
                lines.append(line[1][0])

    print(type(result), result[:1])

    return "\n".join(lines)


def ocr_scanned_pdf(pdf_path):
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    text = []

    for i, page in enumerate(pages):
        temp_img = f"_page_{i}.png"
        page.save(temp_img)
        text.append(ocr_image(temp_img))
        os.remove(temp_img)

    return "\n".join(text)


def extract_digital_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# ---------------- MAIN ----------------
if __name__ == "__main__":

    file_path = "data.jpg"

    if not os.path.exists(file_path):

        print("❌ File not found")
        exit()

    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".png", ".jpg", ".jpeg"]:
        print("\n🖼 OCR on image\n")
        raw_text = ocr_image_text_and_tables(file_path)


    elif ext == ".pdf":
        if is_scanned_pdf(file_path):
            print("\n📄 OCR on scanned PDF\n")
            raw_text = ocr_scanned_pdf(file_path)
        else:
            print("\n📄 Extracting text from digital PDF\n")
            raw_text = extract_digital_pdf(file_path)
    else:
        print("❌ Unsupported file type")
        exit()

    print("\n================ EXTRACTED TEXT ================\n")
    print(clean_text(raw_text))
    print("\n================================================\n")
