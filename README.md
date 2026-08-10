# Subject Extractor

A local tool that takes any photograph containing stickers, graffiti, signage, objects, or people, auto-detects subject boundaries, provides an interactive browser curation UI, and outputs clean, individually-masked transparent PNGs with automated file naming for collage and motion design workflows.

---

## Features

- **Auto-Detection**: MobileSAM (Mobile Segment Anything Model) detects distinct candidate subjects across complex backgrounds.
- **Interactive Curation UI**: Fast web interface to preview, select, discard, or select all candidate masks. Includes a live alpha checkerboard preview and API key configuration.
- **Edge Refinement**: High-quality alpha matting (`rembg` with `u2net`) for crisp edges around fine details, hair, and soft boundaries.
- **Smart Automated Naming**: Multimodal VLM (`moondream2`) + OCR (`pytesseract`) + optional LLM (`deepseek-v4-flash`) pipeline to generate clean, descriptive filenames automatically.
- **Contact Sheet Generation**: Automatically compiles an overview grid image of all extracted subjects for rapid curation.
- **Sequential Memory Management**: Unloads unused neural net models between pipeline stages to fit smoothly within hardware memory limits (e.g., 8GB Apple Silicon).

---

## Architecture & Pipeline

```
[ Input Photo ]
       │
       ▼
 1. SEGMENT     ───► MobileSAM auto-segments candidate subject bounding boxes & masks
       │
       ▼
 2. CURATE      ───► FastAPI Web UI (http://127.0.0.1:8000) for user selection
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
[ Output Directory: PNGs + Contact Sheet + manifest.json ]
```

---

## Prerequisites

### 1. System Dependencies

#### Tesseract OCR
- **macOS**: `brew install tesseract`
- **Ubuntu / Debian**: `sudo apt-get install tesseract-ocr`
- **Windows**: Install via [Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki) and ensure `tesseract` is added to system PATH.

---

## Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/subject-extractor.git
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

Run the main pipeline on any target image:

```bash
python main.py /path/to/your/image.jpg
```

1. The script will analyze the image using MobileSAM.
2. A web browser window will open automatically at `http://127.0.0.1:8000`.
3. Check/uncheck subjects to extract.
4. Click **Extract Selected** to finish processing.
5. Extracted transparent PNGs and `contact_sheet.png` will be saved to `./outputs/<timestamp>/`.

---

## Configuration

Settings can be customized in `config.yaml`:

```yaml
mobilesam_checkpoint: "models/mobile_sam.pt"
deepseek_model: "deepseek-v4-flash"
web_ui_port: 8000
alpha_matting: true
```

---

## License

This project is open source under the [MIT License](LICENSE).
