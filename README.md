# Subject Extractor

A local AI tool that takes any photograph containing stickers, graffiti, signage, objects, or people, auto-detects subject boundaries, provides an interactive browser curation UI (with direct drag-and-drop photo upload), and outputs clean, individually-masked transparent PNGs with automated file naming for collage and motion design workflows.

---

## Features

- **Direct Web UI Photo Upload**: Drag-and-drop any photo directly inside your browser or run standalone via `python main.py`.
- **Auto-Detection**: MobileSAM (Mobile Segment Anything Model) detects distinct candidate subjects across complex backgrounds.
- **Interactive Studio Curation UI**: Fast web interface with zoom/pan inspection (`0.5x` to `5.0x`), manual box drawing mode, live checkerboard toggle, and keyboard shortcuts.
- **Edge Refinement**: High-quality alpha matting (`rembg` with `u2net`) for crisp edges around fine details, hair, and soft boundaries.
- **Smart Automated Naming**: Multimodal VLM (`moondream2`) + OCR (`pytesseract`) + optional LLM (`deepseek-v4-flash`) pipeline to generate clean, descriptive filenames automatically.
- **Contact Sheet Generation**: Automatically compiles an overview grid image of all extracted subjects for rapid curation.
- **Sequential Memory Management**: Unloads unused neural net models between pipeline stages to fit smoothly within hardware memory limits (e.g., 8GB Apple Silicon).
- **Fast/Quality Analysis Modes**: Fast mode uses lightweight foreground extraction for quick single-subject results; Quality mode uses MobileSAM for multiple candidate subjects.

---

## Architecture & Pipeline

```
[ Web UI Drag & Drop Photo Upload ]
                 │
                 ▼
 1. SEGMENT     ───► MobileSAM auto-segments candidate subject bounding boxes & masks
                 │
                 ▼
 2. CURATE      ───► FastAPI Web Studio (http://127.0.0.1:8000) for user selection & box drawing
                 │
                 ▼
 3. EXTRACT     ───► Crop & isolated bounding box alpha masking
                 │
                 ▼
 4. REFINE      ───► rembg alpha matting for edge quality
                 │
                 ▼
 5. LABEL       ───► moondream2 (caption) + Tesseract (OCR) + DeepSeek API (clean slug)
                 │
                 ▼
[ Background Job Queue ] ───► Progress / Cancel / Retry / History / ZIP Export
```

---

## Prerequisites

### System Dependencies

#### Tesseract OCR
- **macOS**: `brew install tesseract`
- **Ubuntu / Debian**: `sudo apt-get install tesseract-ocr`
- **Windows**: Install via [Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki) and ensure `tesseract` is added to system PATH.

---

## Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/wildyonge-dot/subject-extractor.git
   cd subject-extractor
   ```

2. **Create Virtual Environment & Install Dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure API Key (Optional)**
   - Copy `.env.example` to `.env` or set `DEEPSEEK_API_KEY` in your environment.
   - Alternatively, enter your API key directly in the Web UI field during curation.

---

## Usage

### Option 1: Standalone Web Studio Mode (Recommended)
Simply launch `main.py` without arguments:

```bash
python main.py
```

On macOS/Linux, you can also use:

```bash
./start.sh
```
This automatically opens **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser. Drag and drop any photo to begin!

The uploader defaults to **Fast foreground** analysis. Choose **Quality multi-subject** in the upload screen when you need MobileSAM candidate masks.

The upload screen also supports AI/OCR/basic naming, box and polygon prompts, recent-job history, retry/cancel, and ZIP export. The server keeps jobs under `outputs/` and recovers completed manifests when restarted.

### Option 2: Command Line Path Mode
Pass an image path directly:

```bash
python main.py /path/to/your/image.jpg
```

Use `--mode quality` for MobileSAM or `--mode fast` for lightweight foreground extraction:

```bash
python main.py /path/to/your/image.jpg --mode quality
```

For batch processing:

```bash
python auto_extract.py /path/to/images --mode fast --label-mode basic
```

Run the lightweight test suite with:

```bash
python -m unittest discover -s tests
```

---

## License

This project is open source under the [MIT License](LICENSE).
