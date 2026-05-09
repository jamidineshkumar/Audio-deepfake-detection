import joblib
import os
import sys
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# --- STEP 1: PATH & IMPORT FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing.feature_extraction import extract_combined_voice_features
from data_processing.load_audio import get_audio_files

# --- STEP 2: PROTOCOL LOGIC (Must match train_model.py) ---
def get_labels_from_protocol(protocol_path):
    label_map = {}
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                file_id = parts[1]
                key = parts[4]
                label_map[file_id] = 1 if key == "bonafide" else 0
    return label_map

def create_feature_dataset(dataset_path, protocol_path):
    X = []
    y = []
    label_map = get_labels_from_protocol(protocol_path)
    audio_files = get_audio_files(dataset_path)
    print(f"Loading {len(audio_files)} files for evaluation...")

    for file_path in audio_files:
        file_id = os.path.basename(file_path).replace(".flac", "")
        if file_id in label_map:
            features = extract_combined_voice_features(file_path)
            X.append(features)
            y.append(label_map[file_id])
    return np.array(X), np.array(y)

# --- STEP 3: DATA LOADING ---
print("Loading dataset...")
base_path = r"D:\AUDIO\Audio-deepfake-detection\DATASET\LA\LA"
audio_path = os.path.join(base_path, "ASVspoof2019_LA_train", "flac")
protocol_path = os.path.join(base_path, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")

X, y = create_feature_dataset(audio_path, protocol_path)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- STEP 4: LOAD ALL MODELS (Including XGBoost) ---
svm_model = joblib.load("saved_models/svm_model.pkl")
rf_model = joblib.load("saved_models/rf_model.pkl")
lr_model = joblib.load("saved_models/lr_model.pkl")
gb_model = joblib.load("saved_models/gb_model.pkl")
nb_model = joblib.load("saved_models/nb_model.pkl")
ann_model = joblib.load("saved_models/ann_model.pkl")
lin_reg_model = joblib.load("saved_models/lin_reg_model.pkl")
xgb_model = joblib.load("saved_models/xgb_model.pkl") # Added XGBoost

# --- STEP 5: PREDICTIONS ---
svm_predictions = svm_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)
lr_predictions = lr_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)
nb_predictions = nb_model.predict(X_test)
ann_predictions = ann_model.predict(X_test)
lin_reg_predictions = (lin_reg_model.predict(X_test) >= 0.5).astype(int)
xgb_predictions = xgb_model.predict(X_test) # Added XGBoost

# --- STEP 6: TERMINAL OUTPUT ---
models_to_print = [
    ("SVM", svm_predictions),
    ("Random Forest", rf_predictions),
    ("Logistic Regression", lr_predictions),
    ("Gradient Boosting", gb_predictions),
    ("Naive Bayes", nb_predictions),
    ("ANN", ann_predictions),
    ("Linear Regression", lin_reg_predictions),
    ("XGBoost", xgb_predictions) # Added XGBoost
]

for name, preds in models_to_print:
    print(f"\n--- {name} Evaluation ---")
    print("Classification Report:\n")
    print(classification_report(y_test, preds))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, preds))

# --- STEP 7: SAVE TO FILE ---
os.makedirs("results", exist_ok=True)

with open("results/model_results.txt", "w") as f:
    f.write("Model Evaluation Results & Comparison\n")
    f.write("======================================\n\n")
    
    for name, preds in models_to_print:
        f.write(f"--- {name} ---\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, preds))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, preds)))
        f.write("\n\n")

print("\nResults saved in results/model_results.txt")