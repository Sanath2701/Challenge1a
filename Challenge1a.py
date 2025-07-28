#!/usr/bin/env python
# coding: utf-8

# In[6]:


#get_ipython().run_line_magic('pip', 'install pymupdf')


# In[9]:


from langdetect import detect
import fitz  # PyMuPDF
import os

# ---------------------------------
# Detect dominant language in PDF
# ---------------------------------

def detect_pdf_language(pdf_path, max_pages=2):
    try:
        doc = fitz.open(pdf_path)
        text_sample = ""

        for page in doc[:max_pages]:
            text_sample += page.get_text()

        if not text_sample.strip():
            return "unknown"

        lang = detect(text_sample)
        print(f"🌐 Detected language: {lang.upper()}")
        return lang.lower()

    except Exception as e:
        print(f"❌ Language detection error: {e}")
        return "unknown"


# In[2]:


#..................Pre-proccessing...........................
import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output
from PIL import Image, ImageFilter
import io
import unicodedata
import re
from typing import List, Dict

def clean_text(text: str) -> str:
    """Normalize and remove redundant whitespace."""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_ocr_text(text: str) -> str:
    text = re.sub(r'([^\s])\1{2,}', r'\1', text)              # Remove excessive character repeats
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)            # Remove repeated words
    text = re.sub(r'(?<!\w) (\w) (?!\w)', r'\1', text)        # Remove spaced characters
    return text.strip()

def ocr_page_with_layout(page) -> List[Dict]:
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    data = pytesseract.image_to_data(img, lang='eng', output_type=pytesseract.Output.DICT)
    lines = []

    for i in range(len(data["text"])):
        text = clean_ocr_text(data["text"][i])
        if not text.strip():
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])
        font_size = height  # Approximate

        line_info = {
            "text": clean_text(text),
            "x": x,
            "y": y,
            "height": height,
            "fontsize": font_size,
            "font": "OCR",
            "bold": False,
            "italic": False,
            "color": 0,
            "page": page.number + 1,
            "spans_info": []
        }
        lines.append(line_info)

    return lines

def extract_pdf_lines(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    all_lines = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        page_lines = []

        for block in blocks:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                spans = line.get("spans", [])
                visible_text = "".join(s["text"] for s in spans if s.get("text", "").strip())

                if not visible_text.strip():
                    continue

                sizes = [s["size"] for s in spans if s.get("size")]
                fonts = [s["font"] for s in spans if s.get("font")]
                colors = [s["color"] for s in spans if s.get("color") is not None]

                avg_fontsize = sum(sizes) / len(sizes) if sizes else 0
                first_font = fonts[0] if fonts else ""

                page_lines.append({
                    "text": clean_text(visible_text),
                    "x": line["bbox"][0],
                    "y": line["bbox"][1],
                    "height": line["bbox"][3] - line["bbox"][1],
                    "fontsize": avg_fontsize,
                    "font": first_font,
                    "bold": any("bold" in f.lower() or "black" in f.lower() for f in fonts),
                    "italic": any("italic" in f.lower() or "oblique" in f.lower() for f in fonts),
                    "color": colors[0] if colors else 0,
                    "page": page_num,
                    "spans_info": spans,
                })

        # OCR fallback if no text
        if not page_lines:
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            data = pytesseract.image_to_data(img, lang='eng', output_type=Output.DICT)

            for i in range(len(data["text"])):
                text = clean_text(data["text"][i])
                if not text or len(text) <= 2:
                    continue

                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                page_lines.append({
                    "text": text,
                    "x": x,
                    "y": y,
                    "height": h,
                    "fontsize": h,
                    "font": "OCR",
                    "bold": False,
                    "italic": False,
                    "color": 0,
                    "page": page_num,
                    "spans_info": [],
                })

        all_lines.extend(page_lines)

    return all_lines




# In[3]:


#..............Per Page Analysis.................................
from collections import defaultdict, Counter
import numpy as np
from typing import List, Dict

def analyze_pages(lines: List[Dict]) -> Dict[int, Dict]:
    page_stats = defaultdict(lambda: {
        "line_count": 0,
        "font_sizes": [],
        "fonts": [],
        "line_heights": [],
        "y_positions": [],
        "x_positions": [],
        "line_lengths": [],
        "bold_count": 0,
        "italic_count": 0,
        "top_lines": [],
        "bottom_lines": [],
        "heading_candidates": 0,
        "is_probable_table_page": False  # <-- initialize here
    })

    page_line_map = defaultdict(list)

    for line in lines:
        page = line["page"]
        page_stats[page]["line_count"] += 1
        page_stats[page]["font_sizes"].append(line["fontsize"])
        page_stats[page]["fonts"].append(line["font"])
        page_stats[page]["line_heights"].append(line["fontsize"])  # Approx height from font size
        page_stats[page]["y_positions"].append(line["y"])
        page_stats[page]["x_positions"].append(line["x"])
        page_stats[page]["line_lengths"].append(len(line["text"]))
        page_stats[page]["bold_count"] += int(line["bold"])
        page_stats[page]["italic_count"] += int(line["italic"])
        page_line_map[page].append(line)

        # Approximate heading candidate heuristic
        if (
            line["bold"] or line["italic"]
            or line["fontsize"] >= 13  # heuristic threshold
            or len(line["text"].split()) < 12
        ):
            page_stats[page]["heading_candidates"] += 1

    for page, stats in page_stats.items():
        lines_sorted = sorted(page_line_map[page], key=lambda l: l["y"])

        # Capture top 2 lines (header) and bottom 2 lines (footer)
        stats["top_lines"] = [l["text"] for l in lines_sorted[:2]]
        stats["bottom_lines"] = [l["text"] for l in lines_sorted[-2:]]

        # Most common font
        stats["common_fonts"] = Counter(stats["fonts"]).most_common(3)
        stats["avg_font_size"] = round(np.mean(stats["font_sizes"]), 2) if stats["font_sizes"] else 0

        # Most common x (alignment patterns)
        stats["most_common_x"] = Counter(
            [round(x, 0) for x in stats["x_positions"]]
        ).most_common(1)[0][0] if stats["x_positions"] else None

        # Line spacing variance (approximated by y-spacing)
        y_sorted = sorted(stats["y_positions"])
        y_diffs = np.diff(y_sorted)
        stats["line_spacing_variance"] = round(np.var(y_diffs), 2) if len(y_diffs) >= 2 else 0

        # ✅ Add table page flag
        stats["is_probable_table_page"] = (
            stats["line_count"] > 40 and stats["line_spacing_variance"] < 5
        )

    return dict(page_stats)


# In[4]:


from typing import List, Dict
import re

# ---------------------------
# Post-processing Utils
# ---------------------------
def normalize_heading_text(text: str) -> str:
    return re.sub(r'\W+', ' ', text).strip().lower()

def postprocess_title(title: str) -> str:
    def remove_repeated_words(text: str) -> str:
        words = text.split()
        cleaned = [words[0]] if words else []
        for word in words[1:]:
            if word != cleaned[-1]:
                cleaned.append(word)
        return " ".join(cleaned)

    def clean_ocr_artifacts(text: str) -> str:
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'([^\w\s])\1+', r'\1', text)
        text = re.sub(r'(\b\w{1,3}\b)( \1)+', r'\1', text)
        return text.strip()

    title = remove_repeated_words(title)
    title = clean_ocr_artifacts(title)
    title = title.replace('|', 'I') \
                 .replace('“', '"').replace('”', '"') \
                 .replace('‘', "'").replace('’', "'")
    return title.strip()

# ---------------------------
# Title Extraction
# ---------------------------
def extract_title(lines: List[Dict], page_stats: Dict[int, Dict]) -> str:
    # Step 1: Use first page (page 0 or 1)
    page1_lines = [l for l in lines if l["page"] in (0, 1)]
    if not page1_lines:
        return "Untitled"

    # Step 2: Top 80% of the page
    page_height = 842  # A4
    max_y_limit = page_height * 0.8

    # Step 3: Get largest font size threshold
    max_font = max((l["fontsize"] for l in page1_lines), default=0)
    font_threshold = 0.9 * max_font
    title_candidates = [
        l for l in page1_lines
        if l["fontsize"] >= font_threshold
        and len(l["text"].strip()) >= 3
        and l["y"] < max_y_limit
    ]

    # Step 4: Remove repeated headers/footers from other pages
    repeated_texts = set()
    for page, stats in page_stats.items():
        if page in (0, 1):
            continue
        repeated_texts.update(stats.get("top_lines", []))
        repeated_texts.update(stats.get("bottom_lines", []))

    title_candidates = [
        l for l in title_candidates
        if l["text"].strip() not in repeated_texts
    ]

    if not title_candidates:
        return "Untitled"

    # Step 5: Group nearby lines (multi-line title support)
    title_candidates.sort(key=lambda l: l["y"])
    grouped = []
    current_group = []

    for line in title_candidates:
        if not current_group:
            current_group.append(line)
            continue

        prev = current_group[-1]
        same_block = (
            abs(line["y"] - prev["y"]) <= 40 and
            abs(line["x"] - prev["x"]) <= 100
        )

        if same_block:
            current_group.append(line)
        else:
            grouped.append(current_group)
            current_group = [line]

    if current_group:
        grouped.append(current_group)

    if not grouped:
        return "Untitled"

    # Step 6: Pick group with most content
    best_group = max(grouped, key=lambda g: len(" ".join(l["text"] for l in g)))
    title_lines = [l["text"].strip() for l in best_group]
    raw_title = " ".join(title_lines)

    # Step 7: Postprocess title
    cleaned_title = postprocess_title(raw_title)

    # Step 8: Final filter for junk/decorative phrases
    blacklist_phrases = {
        "hope to see you there",
        "welcome",
        "thank you",
        "best wishes",
        "rsvp",
        "please join us",
        "you are invited",
        "see you soon"
    }

    if (
        len(cleaned_title.split()) < 4 or
        normalize_heading_text(cleaned_title) in blacklist_phrases or
        re.fullmatch(r"[A-Z ]{3,}", cleaned_title) or
        re.search(r"http|www|\.com", cleaned_title, re.IGNORECASE)
    ):
        return "Untitled"

    return cleaned_title or "Untitled"


# In[5]:


import re
from functools import lru_cache
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util

# -------------------- MODEL INIT --------------------
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# -------------------- CONFIG --------------------
CONFIDENCE_THRESHOLD = 0.7
HEADING_KEYWORDS = [
    "introduction", "summary", "objectives", "acknowledgement",
    "references", "conclusion", "scope", "background", "purpose", "goals",
    "related work", "discussion", "results", "requirements"
]
SECTION_PATTERN = re.compile(r"^\d+(\.\d+)*\s+")

PLACEHOLDER_PATTERN = re.compile(
    r"(RSVP|DATE|NAME|SIGNATURE|EMAIL|ADDRESS|PHONE|FAX)\b[:\s\-]*$",
    re.IGNORECASE
)

# -------------------- NORMALIZATION --------------------
def normalize_heading_text(text: str) -> str:
    return re.sub(r'\W+', ' ', text).strip().lower()

# -------------------- SEMANTIC CACHE --------------------
@lru_cache(maxsize=512)
def cached_semantic_score(text: str, keywords: tuple = ()) -> float:
    words = text.strip().split()
    if not words or len(words) > 10:
        return 0.0
    emb1 = model.encode(text, convert_to_tensor=True)
    emb2 = model.encode(list(keywords), convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).max().item()

# -------------------- FONT CLUSTER TO HEADING LEVEL --------------------
def get_level_by_font(fontsize: float, font_clusters: List[float]) -> str:
    for i, size in enumerate(font_clusters):
        if abs(fontsize - size) < 0.5:
            return f"H{i + 1}"
    return "H3"

# -------------------- HEADING CANDIDATE CHECK --------------------
def is_heading_candidate(
    line: Dict,
    normalized_title: str,
    page_stats: Dict[int, Dict]
) -> bool:
    text = line["text"].strip()
    if not text:
        return False

    normalized = normalize_heading_text(text)
    page = line["page"]
    stats = page_stats.get(page, {})

    normalized_tops = [normalize_heading_text(t) for t in stats.get("top_lines", [])]
    normalized_bottoms = [normalize_heading_text(t) for t in stats.get("bottom_lines", [])]

    # --- Exclusion checks ---
    if normalized in normalized_tops + normalized_bottoms:
        return False
    if re.fullmatch(r"\d{1,2}[\.\)]", text):
        return False
    if re.fullmatch(r"[\d\s\.\)\(]+", text):
        return False
    if re.search(r":\s*$", text) and len(text.split()) <= 4:
        return False
    if re.match(r".{1,20}:\s*-{3,}", text):
        return False
    if PLACEHOLDER_PATTERN.match(text):
        return False

    # ✅ Title similarity check — penalize but allow
    title_similarity = cached_semantic_score(normalized, (normalized_title,))
    title_penalty = -0.2 if title_similarity > 0.8 else 0.0

    # --- Scoring ---
    avg_font = stats.get("avg_font_size", 10)
    most_common_x = stats.get("most_common_x", 50)

    font_score = min(line["fontsize"] / avg_font, 2.0)
    bold_score = 1.0 if line["bold"] else 0.0
    italic_score = 0.5 if line["italic"] else 0.0
    left_align_score = 1.0 if abs(line["x"] - most_common_x) <= 20 else 0.0
    section_score = 1.0 if SECTION_PATTERN.match(text) else 0.0
    semantic_score = cached_semantic_score(text, tuple(HEADING_KEYWORDS)) if len(text.split()) <= 10 else 0.0

    same_y_count = sum(1 for y in stats.get("y_positions", []) if abs(y - line["y"]) < 8)
    dense_penalty = -0.3 if same_y_count > (stats.get("line_count", 30) / 8) else 0.0
    len_penalty = -0.2 if len(text.split()) > 15 or len(text) > 120 else 0.0

    score = (
        0.25 * font_score +
        0.2 * bold_score +
        0.1 * italic_score +
        0.2 * section_score +
        0.15 * left_align_score +
        0.1 * semantic_score +
        dense_penalty + len_penalty + title_penalty
    )
    score = min(score, 1.5)
    return score >= CONFIDENCE_THRESHOLD

# -------------------- FINAL HEADING EXTRACTION --------------------
def extract_headings(lines: List[Dict], normalized_title: str, page_stats: Dict[int, Dict]) -> List[Dict]:
    headings = []

    all_fonts = [line["fontsize"] for line in lines if line.get("fontsize", 0) > 0]
    font_clusters = sorted(set(all_fonts), reverse=True)[:3]

    for line in lines:
        if is_heading_candidate(line, normalized_title, page_stats):
            text = line["text"].strip()
            page = max(0, line["page"] - 1)  # Ensure non-negative index

            match = SECTION_PATTERN.match(text)
            if match:
                dot_count = match.group(0).count(".")
                level = f"H{min(dot_count + 1, 6)}"
            else:
                level = get_level_by_font(line["fontsize"], font_clusters)

            headings.append({
                "level": level,
                "text": text,
                "page": page
            })

    return headings


# In[12]:


import fitz

FRENCH_GERMAN_TITLE_STOPWORDS = [
    "sommaire", "résumé", "abstract", "inhalt", "zusammenfassung", "verzeichnis"
]

def extract_multilingual_title_fr_de(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    blocks = page.get_text("dict")["blocks"]

    candidate = ""
    max_size = 0

    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                font_size = span["size"]
                font_name = span.get("font", "").lower()

                # Skip empty or long lines
                if not text or len(text) > 150:
                    continue

                # Exclude generic headings like 'Sommaire'
                if text.lower() in FRENCH_GERMAN_TITLE_STOPWORDS:
                    continue

                # Look for bold or large text
                if font_size > max_size and ("bold" in font_name or font_size >= 18):
                    max_size = font_size
                    candidate = text

    return candidate or "Untitled"


# In[13]:


from pdf2image import convert_from_path
from pytesseract import image_to_string

FRENCH_GERMAN_SKIPWORDS = ["sommaire", "résumé", "abstract", "inhalt", "verzeichnis"]

def extract_headings_fr_de(pdf_path, skip_pages=[], lang="fra+deu"):
    doc = fitz.open(pdf_path)
    images = convert_from_path(pdf_path, dpi=200)
    headings = []

    for page_num, page in enumerate(doc):
        if page_num in skip_pages:
            continue

        blocks = page.get_text("dict")["blocks"]
        found_text = False

        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_text = ""
                sizes = []
                fonts = set()

                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue

                    if text.lower() in FRENCH_GERMAN_SKIPWORDS:
                        continue

                    line_text += text + " "
                    sizes.append(span["size"])
                    fonts.add(span.get("font", "").lower())

                line_text = line_text.strip()
                if not line_text or len(line_text) > 120:
                    continue

                found_text = True
                avg_size = sum(sizes) / len(sizes)
                font_name = " ".join(fonts)
                is_heading_style = ("bold" in font_name or "italic" in font_name)

                if avg_size >= 12 and is_heading_style:
                    level = determine_heading_level(avg_size)
                    headings.append({
                        "page": page_num + 1,
                        "text": line_text,
                        "level": level,
                        "method": "font"
                    })

        # OCR fallback
        if not found_text:
            print(f"[OCR] Page {page_num + 1} fallback")
            ocr_text = image_to_string(images[page_num], lang=lang)
            for line in ocr_text.split("\n"):
                line = line.strip()
                if line and len(line) < 100 and line.lower() not in FRENCH_GERMAN_SKIPWORDS:
                    headings.append({
                        "page": page_num + 1,
                        "text": line,
                        "level": "H2 (OCR)",
                        "method": "ocr"
                    })

    return headings


# In[15]:


def determine_heading_level(size):
    if size >= 18:
        return "H1"
    elif size >= 14:
        return "H2"
    elif size >= 12:
        return "H3"
    else:
        return "H4"


# In[6]:


import os
import json
from typing import List, Dict

def save_outline_to_json(title: str, headings: List[Dict], path: str):
    output_dir = os.path.dirname(path)
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "title": title,
            "outline": headings
        }, f, indent=2, ensure_ascii=False)



# In[18]:


from pytesseract import image_to_string
from pdf2image import convert_from_path
from collections import defaultdict
import re

def extract_headings_multilingual(pdf_path, skip_pages=[], lang='fra+deu'):
    images = convert_from_path(pdf_path, dpi=200)
    headings = []

    for page_num, image in enumerate(images):
        if page_num in skip_pages:
            continue

        print(f"🔎 OCR on Page {page_num + 1}")
        text = image_to_string(image, lang=lang)
        lines = text.split("\n")

        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                continue

            # Heuristic for heading-like lines: short, capitalized, ends with no period
            if (1 <= len(clean_line.split()) <= 10 and
                clean_line[0].isupper() and
                not clean_line.endswith(".")):

                level = estimate_heading_level(clean_line)
                headings.append({
                    "text": clean_line,
                    "page_num": page_num + 1,
                    "level": level,
                    "source": "ocr"
                })

    return headings


def estimate_heading_level(text):
    # Basic heuristic: longer = lower level
    word_count = len(text.split())
    if word_count <= 3:
        return 1
    elif word_count <= 6:
        return 2
    else:
        return 3


# In[10]:


from langdetect import detect
import os

def detect_pdf_language(pdf_path, max_pages=2):
    import fitz
    try:
        doc = fitz.open(pdf_path)
        text_sample = ""

        for page in doc[:max_pages]:
            text_sample += page.get_text()

        if not text_sample.strip():
            return "unknown"

        lang = detect(text_sample)
        print(f"🌐 Detected language: {lang.upper()}")
        return lang.lower()

    except Exception as e:
        print(f"❌ Language detection error: {e}")
        return "unknown"

def main(pdf_path: str, output_dir: str = "Output"):
    # Step 0: Detect language
    detected_lang = detect_pdf_language(pdf_path)

    # Prepare output file name
    filename_without_ext = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{filename_without_ext}.json")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------
    # Case 1: English or Unknown Language
    # ------------------------------------------
    if detected_lang in ["en", "unknown"]:
        print("🔍 Language is English or unknown. Running English pipeline...")

        # Step 1: Extract lines from PDF
        print("📄 Extracting lines from PDF...")
        lines = extract_pdf_lines(pdf_path)

        # Step 2: Analyze layout stats
        print("📊 Analyzing page statistics...")
        page_stats = analyze_pages(lines)

        # Step 3: Extract title from page 1
        print("🏷️ Extracting title...")
        title = extract_title(lines, page_stats)
        normalized_title = normalize_heading_text(title)

        # Step 4: Extract structured headings
        print("🧠 Detecting headings...")
        headings = extract_headings(lines, normalized_title, page_stats)

        # Step 5: Save output
        print(f"💾 Saving output to: {output_path}")
        save_outline_to_json(title, headings, path=output_path)

        print(f"✅ Done! Extracted title and {len(headings)} headings.\n")

    # ------------------------------------------
    # Case 2: Detected Multilingual PDF
    # ------------------------------------------
    else:
        print(f"🌐 Detected non-English language: {detected_lang.upper()}. Running multilingual + OCR pipeline...")

        # Step 1: Extract multilingual headings (OCR fallback included)
        multilingual_headings = extract_headings_multilingual(pdf_path, skip_pages=[])

        # Use a placeholder for title
        title = multilingual_headings[0]['text'] if multilingual_headings else "Untitled Document"

        # Step 2: Save output
        print(f"💾 Saving multilingual output to: {output_path}")
        save_outline_to_json(title, headings, path=output_path)


        print(f"✅ Done! Extracted {len(multilingual_headings)} multilingual headings.\n")


# In[19]:


import os

if __name__ == "__main__":
    input_folder = "app/Input"
    output_folder = "app/Output"

    if not os.path.exists(input_folder):
        print(f"❌ Input folder '{input_folder}' not found.")
        exit(1)

    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("⚠️ No PDF files found in 'Input' folder.")
    else:
        for pdf_file in pdf_files:
            print(f"\n📄 Processing {pdf_file}...\n")
            pdf_path = os.path.join(input_folder, pdf_file)
            main(pdf_path, output_dir=output_folder)

