Adobe India Hackathon 2025 – Challenge 1A: 📘 PDF Title & Heading Extractor
A powerful tool to automatically extract the document title and structured section headings (H1–H4) from English, French, and German PDFs using a hybrid approach of layout heuristics, OCR, and semantic intelligence. Output is a clean JSON file suitable for search, summarization, or TOC generation.

🚀 Features
Feature	Included	Description
🔤 Multilingual support (EN, FR, DE)	✅	Extracts from English, French, and German PDFs automatically
🏷️ Title extraction from Page 1	✅	Dynamically detects and merges top-font lines
🧠 Heading detection with scoring engine	✅	Combines font, layout, regex & semantic cues
⚙️ Semantic filtering with caching	✅	SentenceTransformer + @lru_cache for speed
📐 Adaptive layout analysis	✅	Thresholds adapt to each document’s layout
📖 Heading levels (H1–H6 via regex)	✅	Supports numbered sections like 1.2, 2.1.3
📉 Table page skipping	✅	Auto-skips dense data pages
💾 Output as JSON	✅	Output/<filename>.json format
🧾 OCR fallback for scanned PDFs	✅	Tesseract-powered OCR for image-based PDFs
🔄 Batch processing	✅	Processes all PDFs in Input/ automatically

🌍 Multilingual Intelligence (New!)
Language	Detection Method	Extraction Strategy
English	langdetect + layout	Heuristics + font stats + semantic scoring
French	Auto-detected	Tesseract OCR + bold/italic + font size
German	Auto-detected	Tesseract OCR + heading pattern match

⚠️ For scanned or low-text PDFs, OCR ensures fallback accuracy.

🧠 Strategy Breakdown
📄 Preprocessing
Extracts visible lines with font, bold/italic, x/y layout

Uses Tesseract OCR when needed (image-based PDFs)

📊 Page-Level Stats
Captures line density, spacing variance, common font sizes

Flags table-heavy pages to reduce false positives

🏷️ Title Detection
Finds largest, top-most font block (English, FR, DE)

De-duplicates headers and footers

Cleans repeated words and artifacts (e.g., "“”, ‘’, ||")

🧠 Heading Extraction Logic
Uses font size ratios, alignment, bold/italic emphasis

Detects sectioning patterns like 1.2, 2.1.3

Semantic similarity via sentence-transformers for short headings

OCR-based heading guess for multilingual documents