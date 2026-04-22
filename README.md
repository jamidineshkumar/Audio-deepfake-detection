# 🎙️ Audio Deepfake Detection using MFCC and Machine Learning

## 📌 Overview

This project focuses on detecting AI-generated (deepfake) audio using classical machine learning techniques. Speech signals are processed to extract **MFCC (Mel Frequency Cepstral Coefficients)** features, which are then used to train multiple classifiers to distinguish between:

* **Bonafide** → Real human speech
* **Spoof** → AI-generated or manipulated audio

---

## 📂 Dataset

This project uses the **ASVspoof 2019 Logical Access (LA)** dataset, a standard benchmark for voice spoofing detection.

🔗 Download from:
https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset

### 📁 Folder Structure (after download)

```
DATASET/
└── LA
    ├── ASVspoof2019_LA_train
    ├── ASVspoof2019_LA_dev
    └── ASVspoof2019_LA_eval
```

---

## 🏗️ Project Structure

```
audio-deepfake-detection/
│
├── app/                  # Frontend + backend (web interface)
├── data_processing/      # Feature extraction (MFCC)
├── models/               # Training & evaluation scripts
├── notebooks/            # Experimentation (Jupyter notebooks)
├── utils/                # Helper functions
├── results/              # Evaluation outputs
├── saved_models/         # Trained models (ignored in Git)
│
├── config.yaml
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/NirmitSingh-main/audio-deepfake-detection.git
cd audio-deepfake-detection
pip install -r requirements.txt
```

---

## 🧠 Models Trained

The following machine learning models have been trained using MFCC features:

* Support Vector Machine (**SVM**)
* Random Forest (**RF**)
* Logistic Regression (**LR**)
* Linear Regression
* Gaussian Naive Bayes (**NB**)
* Gradient Boosting (**GB**)
* Artificial Neural Network (**ANN**)

📁 Saved in:

```
saved_models/
```

> ⚠️ Note: Trained model files (`.pkl`) are not included in the repository due to size constraints. You need to train them locally.

---

## 🚀 Train the Models

```bash
python -m models.train_model
```

This will:

* Extract MFCC features
* Train all models
* Save them in `saved_models/`

---

## 📊 Evaluate Models

```bash
python -m models.evaluate_model
```

### Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Results are stored in:

```
results/model_results.txt
```

---

## 🌐 Run the Application

```bash
python app/backend.py
```

---

## ⚠️ Important Notes

* Ensure dataset is placed correctly in `DATASET/`
* Models must be trained before running the backend
* `.pkl` files are ignored via `.gitignore`

---

## 🛠️ Technologies Used

* Python
* Librosa
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

---

## 🔮 Future Improvements

* Deep learning models (CNN, LSTM)
* Real-time audio deepfake detection
* Deployment (Flask / FastAPI / Cloud)
* Model optimization & feature engineering

---

## 📌 Author

**Nirmit Singh**

---

## ⭐ Contributing

Pull requests are welcome. For major changes, please open an issue first.

---
