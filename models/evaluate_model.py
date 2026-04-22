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
lr_model = joblib.load("saved_models/lr_model.pkl")
gb_model = joblib.load("saved_models/gb_model.pkl")
nb_model = joblib.load("saved_models/nb_model.pkl")
ann_model = joblib.load("saved_models/ann_model.pkl")
lin_reg_model = joblib.load("saved_models/lin_reg_model.pkl")

svm_predictions = svm_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)
lr_predictions = lr_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)
nb_predictions = nb_model.predict(X_test)
ann_predictions = ann_model.predict(X_test)
lin_reg_predictions = (lin_reg_model.predict(X_test) >= 0.5).astype(int)

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

print("\n--- Logistic Regression Evaluation ---")
print("Classification Report:\n")
print(classification_report(y_test, lr_predictions))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, lr_predictions))

print("\n--- Gradient Boosting Evaluation ---")
print("Classification Report:\n")
print(classification_report(y_test, gb_predictions))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, gb_predictions))

print("\n--- Naive Bayes Evaluation ---")
print("Classification Report:\n")
print(classification_report(y_test, nb_predictions))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, nb_predictions))

print("\n--- ANN Evaluation ---")
print("Classification Report:\n")
print(classification_report(y_test, ann_predictions))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, ann_predictions))

print("\n--- Linear Regression Evaluation ---")
print("Classification Report:\n")
print(classification_report(y_test, lin_reg_predictions))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, lin_reg_predictions))


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

    f.write("--- Logistic Regression ---\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, lr_predictions))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, lr_predictions)))
    f.write("\n\n")

    f.write("--- Gradient Boosting ---\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, gb_predictions))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, gb_predictions)))
    f.write("\n\n")

    f.write("--- Naive Bayes ---\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, nb_predictions))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, nb_predictions)))
    f.write("\n\n")

    f.write("--- ANN ---\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, ann_predictions))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, ann_predictions)))
    f.write("\n\n")

    f.write("--- Linear Regression ---\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, lin_reg_predictions))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, lin_reg_predictions)))
    f.write("\n\n")

print("\nResults saved in results/model_results.txt")
