import os
import sys
import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- STEP 1: PATH FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing.feature_extraction import extract_combined_voice_features
from data_processing.load_audio import get_audio_files

# --- STEP 2: PROTOCOL LOGIC ---
def get_labels_from_protocol(protocol_path):
    """Reads the official ASVspoof protocol to get 100% accurate labels"""
    label_map = {}
    print(f"Loading protocol: {protocol_path}")
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                file_id = parts[1] # e.g., LA_T_1000137
                key = parts[4]     # 'bonafide' or 'spoof'
                label_map[file_id] = 1 if key == "bonafide" else 0
    return label_map

def create_feature_dataset(dataset_path, protocol_path):
    X = []
    y = []
    
    label_map = get_labels_from_protocol(protocol_path)
    audio_files = get_audio_files(dataset_path)
    
    print(f"Total files found in folder: {len(audio_files)}")

    for file_path in audio_files:
        # Extract filename without .flac
        file_id = os.path.basename(file_path).replace(".flac", "")
        
        if file_id in label_map:
            label = label_map[file_id]
            features = extract_combined_voice_features(file_path)
            X.append(features)
            y.append(label)
        
        # Progress tracker for your 25,000 files
        if len(X) % 100 == 0 and len(X) > 0:
            print(f"Processed {len(X)}/{len(audio_files)} files...")
        
    return np.array(X), np.array(y)

# --- STEP 3: EXECUTION ---
print("Initializing Data Pipeline...")

# Paths based on your Screenshot (176)
base_path = r"D:\AUDIO\Audio-deepfake-detection\DATASET\LA\LA"
train_audio_dir = os.path.join(base_path, "ASVspoof2019_LA_train", "flac")
protocol_path = os.path.join(base_path, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")

X, y = create_feature_dataset(train_audio_dir, protocol_path)

if X.size == 0:
    print("Error: No data loaded. Check if filenames in protocol match .flac files.")
    sys.exit()

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)} (0=Fake, 1=Real)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# TRAINING SECTION (All models + XGBoost)
# ---------------------------

# 1. SVM
print("\nTraining SVM...")
svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
svm.fit(X_train, y_train)
print("SVM Accuracy:", accuracy_score(y_test, svm.predict(X_test)))

# 2. RANDOM FOREST
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
rf.fit(X_train, y_train)
print("Random Forest Accuracy:", accuracy_score(y_test, rf.predict(X_test)))

# 3. LOGISTIC REGRESSION
print("Training Logistic Regression...")
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
lr.fit(X_train, y_train)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr.predict(X_test)))

# 4. GRADIENT BOOSTING
print("Training Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100)
gb.fit(X_train, y_train)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb.predict(X_test)))

# 5. NAIVE BAYES
print("Training Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train, y_train)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb.predict(X_test)))

# 6. ANN
print("Training ANN...")
ann = make_pipeline(StandardScaler(), MLPClassifier(max_iter=1000, hidden_layer_sizes=(100,)))
ann.fit(X_train, y_train)
print("ANN Accuracy:", accuracy_score(y_test, ann.predict(X_test)))

# 7. LINEAR REGRESSION
print("Training Linear Regression...")
lin_reg = make_pipeline(StandardScaler(), LinearRegression())
lin_reg.fit(X_train, y_train)
lin_reg_preds = (lin_reg.predict(X_test) >= 0.5).astype(int)
print("Linear Regression Accuracy:", accuracy_score(y_test, lin_reg_preds))

# 8. XGBOOST
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_model.predict(X_test)))

# --- STEP 4: SAVE ---
os.makedirs("saved_models", exist_ok=True)
joblib.dump(svm, "saved_models/svm_model.pkl")
joblib.dump(rf, "saved_models/rf_model.pkl")
joblib.dump(lr, "saved_models/lr_model.pkl")
joblib.dump(gb, "saved_models/gb_model.pkl")
joblib.dump(nb, "saved_models/nb_model.pkl")
joblib.dump(ann, "saved_models/ann_model.pkl")
joblib.dump(lin_reg, "saved_models/lin_reg_model.pkl")
joblib.dump(xgb_model, "saved_models/xgb_model.pkl")

print("\nSuccess! All 8 models saved in 'saved_models' folder.")