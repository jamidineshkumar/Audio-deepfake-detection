import os
import sys
import joblib
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import shutil

# Add project root to path to fix ModuleNotFoundError
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from data_processing.feature_extraction import extract_mfcc

app = FastAPI(title="Audio Deepfake Detection API")

# Load models
# Note: Paths are relative to the project root
SVM_MODEL_PATH = "saved_models/svm_model.pkl"
RF_MODEL_PATH = "saved_models/rf_model.pkl"

def load_models():
    # Try absolute paths if relative fail (for different working directories)
    svm_path = SVM_MODEL_PATH
    rf_path = RF_MODEL_PATH
    
    if not os.path.exists(svm_path):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        svm_path = os.path.join(BASE_DIR, SVM_MODEL_PATH)
        rf_path = os.path.join(BASE_DIR, RF_MODEL_PATH)

    try:
        svm = joblib.load(svm_path)
        rf = joblib.load(rf_path)
        return svm, rf
    except Exception as e:
        print(f"Error loading models from {svm_path}: {e}")
        return None, None

svm_model, rf_model = load_models()
if svm_model and rf_model:
    print("✓ Models (SVM & Random Forest) loaded successfully.")
else:
    print("✗ Error: One or more models could not be loaded. Please check the 'saved_models' directory.")

@app.get("/", response_class=HTMLResponse)
async def root():
    # Try multiple possible paths for frontend.html
    possible_paths = [
        "app/frontend.html",
        os.path.join(os.path.dirname(__file__), "frontend.html"),
        "frontend.html"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                continue
                
    # Fallback if file not found
    return """
    <html>
        <body style="background: #0f172a; color: white; display: flex; align-items: center; justify-content: center; height: 100vh; font-family: sans-serif; text-align: center;">
            <div>
                <h1 style="color: #818cf8;">Audio API is running</h1>
                <p style="color: #94a3b8;">Frontend file (frontend.html) not found in expected locations.</p>
                <p style="color: #94a3b8;">Models loaded: """ + str(svm_model is not None) + """</p>
            </div>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not svm_model or not rf_model:
        raise HTTPException(status_code=500, detail="Models not loaded correctly on server.")

    # Save uploaded file temporarily
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract features
        features = extract_mfcc(file_path)
        features = features.reshape(1, -1)
        
        # Get predictions
        svm_pred = svm_model.predict(features)[0]
        rf_pred = rf_model.predict(features)[0]
        
        # Label mapping from create_dataset.py: 1 if label == "spoof" else 0
        # 1: Fake, 0: Real
        
        res = {
            "filename": file.filename,
            "predictions": {
                "svm": "Fake" if svm_pred == 1 else "Real",
                "random_forest": "Fake" if rf_pred == 1 else "Real"
            },
            "raw_output": {
                "svm": int(svm_pred),
                "random_forest": int(rf_pred)
            }
        }
        
        return JSONResponse(content=res)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
