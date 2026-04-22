import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from data_processing.create_dataset import create_feature_dataset


print("Loading dataset...")

X, y = create_feature_dataset("DATASET")

print("Dataset shape:", X.shape)


# ---------------------------
# Train Test Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)


# ---------------------------
# SVM MODEL
# ---------------------------

print("\nTraining SVM model...")

svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))

svm.fit(X_train, y_train)

svm_predictions = svm.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)

print("SVM Accuracy:", svm_accuracy)


# ---------------------------
# RANDOM FOREST MODEL
# ---------------------------

print("\nTraining Random Forest model...")

rf = RandomForestClassifier(n_estimators=100, class_weight="balanced")

rf.fit(X_train, y_train)

rf_predictions = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("Random Forest Accuracy:", rf_accuracy)


# ---------------------------
# LOGISTIC REGRESSION MODEL
# ---------------------------

print("\nTraining Logistic Regression model...")

lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))

lr.fit(X_train, y_train)

lr_predictions = lr.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_predictions)

print("Logistic Regression Accuracy:", lr_accuracy)


# ---------------------------
# GRADIENT BOOSTING MODEL
# ---------------------------

print("\nTraining Gradient Boosting model...")

gb = GradientBoostingClassifier(n_estimators=100)

gb.fit(X_train, y_train)

gb_predictions = gb.predict(X_test)

gb_accuracy = accuracy_score(y_test, gb_predictions)

print("Gradient Boosting Accuracy:", gb_accuracy)


# ---------------------------
# NAIVE BAYES MODEL
# ---------------------------

print("\nTraining Naive Bayes model...")

nb = GaussianNB()

nb.fit(X_train, y_train)

nb_predictions = nb.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_predictions)

print("Naive Bayes Accuracy:", nb_accuracy)


# ---------------------------
# ANN MODEL
# ---------------------------

print("\nTraining ANN model...")

ann = make_pipeline(StandardScaler(), MLPClassifier(max_iter=1000, hidden_layer_sizes=(100,)))

ann.fit(X_train, y_train)

ann_predictions = ann.predict(X_test)

ann_accuracy = accuracy_score(y_test, ann_predictions)

print("ANN Accuracy:", ann_accuracy)


# ---------------------------
# LINEAR REGRESSION MODEL
# ---------------------------

print("\nTraining Linear Regression model...")

lin_reg = make_pipeline(StandardScaler(), LinearRegression())

lin_reg.fit(X_train, y_train)

# Threshold target scores to generate discrete classes
lin_reg_predictions_cont = lin_reg.predict(X_test)
lin_reg_predictions = (lin_reg_predictions_cont >= 0.5).astype(int)

lin_reg_accuracy = accuracy_score(y_test, lin_reg_predictions)

print("Linear Regression Accuracy (thresholded):", lin_reg_accuracy)


# ---------------------------
# Save best model
# ---------------------------

joblib.dump(svm, "saved_models/svm_model.pkl")
joblib.dump(rf, "saved_models/rf_model.pkl")
joblib.dump(lr, "saved_models/lr_model.pkl")
joblib.dump(gb, "saved_models/gb_model.pkl")
joblib.dump(nb, "saved_models/nb_model.pkl")
joblib.dump(ann, "saved_models/ann_model.pkl")
joblib.dump(lin_reg, "saved_models/lin_reg_model.pkl")

print("\nModels saved in saved_models folder.")