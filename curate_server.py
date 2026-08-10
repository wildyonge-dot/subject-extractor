import os

from fastapi import FastAPI, UploadFile, File, HTTPException
import time
import shutil
import json
from segment import run_segmentation
from extract import extract_subjects
from refine import refine_edges
from label import label_subjects
from main import create_contact_sheet

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Will be set dynamically by the CLI
SESSION_DIR = None
MASKS_DATA = []
SELECTION_EVENT = threading.Event()
SELECTED_IDS = []

class Selection(BaseModel):
    selected_ids: list[str]
    api_key: str = None

class BoxPrompt(BaseModel):
    bbox: list[int]

@app.get("/api/masks")
def get_masks():
    return {"masks": MASKS_DATA}

@app.post("/api/prompt_box")
def prompt_box(box_prompt: BoxPrompt):
    x, y, w, h = box_prompt.bbox
    new_mask_id = f"custom_box_{len(MASKS_DATA) + 1}"
    new_mask = {
        "id": new_mask_id,
        "thumb_file": "test_image.jpg", # We can use test_image or mock it. Wait, the frontend uses /outputs/test_image.jpg. I'll just put a placeholder.
        "area": w * h,
        "bbox": [x, y, w, h]
    }
    MASKS_DATA.append(new_mask)
    return {"status": "ok", "mask": new_mask}

@app.post("/api/submit")
def submit_selection(selection: Selection):
    global SELECTED_IDS, API_KEY, SESSION_DIR, MASKS_DATA
    if selection.api_key and len(selection.api_key.strip()) < 5:
        raise HTTPException(status_code=400, detail="Invalid API key format")
        
    SELECTED_IDS = selection.selected_ids
    API_KEY = selection.api_key
    SELECTION_EVENT.set()
    
    # Run the rest of the pipeline
    if SESSION_DIR:
        source_path = str(SESSION_DIR / "source.jpg")
        # In CLI mode, the source image might not be source.jpg. 
        # But wait, in CLI mode, maybe we don't want to run this here?
        # Let's run it only if source.jpg exists (standalone mode)
        if os.path.exists(source_path):
            extracted = extract_subjects(source_path, SELECTED_IDS, MASKS_DATA, str(SESSION_DIR))
            refined = refine_edges(extracted, str(SESSION_DIR))
            final = label_subjects(refined, str(SESSION_DIR), api_key=API_KEY)
            create_contact_sheet(final, str(SESSION_DIR))
            
            run_id = SESSION_DIR.name
            return {
                "status": "ok", 
                "message": "Extraction complete!", 
                "contact_sheet": f"/outputs/{run_id}/contact_sheet.png"
            }
            
    return {"status": "ok", "message": "Selection received. You can close this window."}

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    global SESSION_DIR, MASKS_DATA, SELECTED_IDS
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = BASE_DIR / "outputs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    SESSION_DIR = output_dir
    
    source_path = output_dir / "source.jpg"
    with open(source_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    masks = run_segmentation(str(source_path), str(output_dir))
    
    # Update thumb paths to be relative to outputs folder
    for m in masks:
        if not m['thumb_file'].startswith(run_id):
            m['thumb_file'] = f"{run_id}/{m['thumb_file']}"
            
    MASKS_DATA = masks
    SELECTED_IDS = []
    
    return {
        "status": "ok",
        "run_id": run_id,
        "image_url": f"/outputs/{run_id}/source.jpg",
        "masks": MASKS_DATA
    }

@app.get("/", response_class=HTMLResponse)
def index():
    with open(BASE_DIR / "static" / "curate.html", "r") as f:
        return f.read()

def start_server(output_dir, masks_data, port=8000):
    global SESSION_DIR, MASKS_DATA, SELECTED_IDS, API_KEY
    SESSION_DIR = Path(output_dir).resolve()
    run_id = SESSION_DIR.name
    for m in masks_data:
        if not m['thumb_file'].startswith(run_id):
            m['thumb_file'] = f"{run_id}/{m['thumb_file']}"
    MASKS_DATA = masks_data
    SELECTED_IDS = []
    API_KEY = None
    SELECTION_EVENT.clear()
    
    # Mount the static and outputs directories
    
    

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    
    # Run the server in a separate thread so we can wait for the event
    def serve():
        server.run()
        
    thread = threading.Thread(target=serve)
    thread.daemon = True
    thread.start()
    
    print(f"Curation UI available at: http://127.0.0.1:{port}")
    print("Waiting for your selection in the browser...")
    
    # Wait for the user to submit
    SELECTION_EVENT.wait()
    
    print(f"Received {len(SELECTED_IDS)} selected subjects. Continuing pipeline...")
    # Give the server a moment to return the response to the client
    import time
    time.sleep(1)
    
    # Cleanly shutdown the server
    server.should_exit = True
    
    return SELECTED_IDS, API_KEY
