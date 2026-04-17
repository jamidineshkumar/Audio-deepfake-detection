# Audio Deepfake Detection using MFCC and Machine Learning

## Overview

This project detects **AI-generated (deepfake) audio** using machine learning.
Speech signals are processed to extract **MFCC (Mel Frequency Cepstral Coefficients)** features, which are then used to train classifiers that distinguish between **real (bonafide)** and **fake (spoof)** audio.

## Dataset

This project uses the **ASVspoof 2019 Logical Access (LA) dataset**, a standard dataset used in research for voice spoofing detection.

Download the dataset from:
https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset

After downloading, place it in the project folder like this:

```
DATASET/
└── LA
    ├── ASVspoof2019_LA_train
    ├── ASVspoof2019_LA_dev
    └── ASVspoof2019_LA_eval
```

The dataset contains:

* **bonafide** → real human speech
* **spoof** → AI-generated or manipulated speech

## Project Structure

```
audio-deepfake-detection
│
├── app
├── data_processing
├── models
├── notebooks
├── utils
├── results
├── saved_models
│
├── config.yaml
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```
git clone https://github.com/NirmitSingh-main/audio-deepfake-detection.git
cd audio-deepfake-detection
```

Install dependencies:

```
pip install -r requirements.txt
```

## Train the Model

```
python -m models.train_model
```

This will:

* Extract MFCC features
* Train **SVM** and **Random Forest** models
* Save trained models in `saved_models/`

## Evaluation

```
python -m models.evaluate_model
```

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

## Technologies Used

* Python
* Librosa
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

## Future Improvements

* Deep learning models (CNN/LSTM)
* Real-time deepfake detection
* Web deployment
