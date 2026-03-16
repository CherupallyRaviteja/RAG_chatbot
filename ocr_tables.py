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
from paddleocr import PaddleOCR

# -------- SILENCE ALL LOGS --------
POPPLER_PATH = None   # set path if poppler is not in PATH (Windows)
LANG = "en"

# ---------------- OCR INIT ----------------
ocr = PaddleOCR(
    lang=LANG,
    use_angle_cls=True, # ✅ valid in all recent versions
)

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


def _format_row(row):
    if len(row) >= 3:
        return " | ".join(row)   # table-like
    else:
        return " ".join(row)     # normal text

if __name__ == "__main__":
    path="ocr_image.jpeg"
    text = ocr_image_text_and_tables(path)
    print(text)