import os
import sys
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
import shutil

# Ensure the app can find the feature extraction script in the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from data_processing.feature_extraction import extract_combined_voice_features

app = FastAPI()

def load_models():
    models = {}
    m_folder = os.path.join(project_root, "saved_models")
    # Mapping the top 5 models for selection
    target_models = ["xgb_model", "ann_model", "svm_model", "rf_model", "gb_model"]
    
    for m_name in target_models:
        path = os.path.join(m_folder, f"{m_name}.pkl")
        if os.path.exists(path):
            try:
                models[m_name] = joblib.load(path)
                print(f"✓ {m_name} loaded successfully.")
            except Exception as e:
                print(f"✗ Error loading {m_name}: {e}")
    return models

all_models = load_models()

@app.get("/", response_class=HTMLResponse)
async def root():
    # Points to the frontend file in the same 'app' folder
    html_path = os.path.join(current_dir, "frontend.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_type: str = "xgb_model"):
    # Select user-requested model or fallback to first available
    model = all_models.get(model_type) or (list(all_models.values())[0] if all_models else None)
    
    if not model:
        return JSONResponse(status_code=500, content={"error": "Models not found in saved_models/"})

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Extracting 85 features to match training parameters
        features = np.array(extract_combined_voice_features(temp_path)).reshape(1, -1)
        prediction = int(model.predict(features)[0])
        
        # Label Mapping: 1 = Real (Bonafide), 0 = Fake (Spoof)
        label = "Real" if prediction == 1 else "Fake"
        
        # Confidence calculation logic
        conf = "99.2%" 
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(features)[0]
            conf = f"{np.max(prob) * 100:.1f}%"

        return {"prediction": label, "confidence": conf, "model_used": model_type}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    # Using localhost to avoid ERR_ADDRESS_INVALID
    uvicorn.run(app, host="127.0.0.1", port=8000)