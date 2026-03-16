import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from data_processing.create_dataset import create_feature_dataset

print("Loading dataset...")

X, y = create_feature_dataset("DATASET")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

import os

svm_model = joblib.load("saved_models/svm_model.pkl")
rf_model = joblib.load("saved_models/rf_model.pkl")

svm_predictions = svm_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

print("\n--- SVM Evaluation ---")
print("Classification Report:\n")
print(classification_report(y_test, svm_predictions))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, svm_predictions))

print("\n--- Random Forest Evaluation ---")
print("Classification Report:\n")
print(classification_report(y_test, rf_predictions))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, rf_predictions))

os.makedirs("results", exist_ok=True)

with open("results/model_results.txt", "w") as f:
    f.write("Model Evaluation Results & Comparison\n")
    f.write("======================================\n\n")
    
    f.write("--- Support Vector Machine (SVM) ---\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, svm_predictions))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, svm_predictions)))
    f.write("\n\n")
    
    f.write("--- Random Forest (RF) ---\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, rf_predictions))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, rf_predictions)))
    f.write("\n\n")

print("\nResults saved in results/model_results.txt")
