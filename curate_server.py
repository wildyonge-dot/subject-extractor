import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

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

@app.get("/api/masks")
def get_masks():
    return {"masks": MASKS_DATA}

@app.post("/api/submit")
def submit_selection(selection: Selection):
    global SELECTED_IDS, API_KEY
    if selection.api_key and len(selection.api_key.strip()) < 5:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid API key format")
        
    SELECTED_IDS = selection.selected_ids
    API_KEY = selection.api_key
    SELECTION_EVENT.set()
    return {"status": "ok", "message": "Selection received. You can close this window."}

@app.get("/", response_class=HTMLResponse)
def index():
    with open(BASE_DIR / "static" / "curate.html", "r") as f:
        return f.read()

def start_server(output_dir, masks_data, port=8000):
    global SESSION_DIR, MASKS_DATA, SELECTED_IDS, API_KEY
    SESSION_DIR = Path(output_dir).resolve()
    MASKS_DATA = masks_data
    SELECTED_IDS = []
    API_KEY = None
    SELECTION_EVENT.clear()
    
    # Mount the static and outputs directories
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    app.mount("/outputs", StaticFiles(directory=str(SESSION_DIR)), name="outputs")

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
