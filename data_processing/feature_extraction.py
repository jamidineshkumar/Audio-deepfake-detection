import librosa
import numpy as np


# ==============================================================================
# FEATURE 1: MFCC (Mel Frequency Cepstral Coefficients)
# The standard feature for speech processing & anti-spoofing.
# Models the vocal tract shape and phonetic content of speech.
# Output: 1D mean vector of shape (n_mfcc,) = (40,)
# ==============================================================================

def extract_mfcc(file_path, n_mfcc=40):
    """
    Extract mean MFCC features.
    Standard feature for voice tasks - models the vocal tract filter.
    Used by: SVM, Random Forest, Logistic Regression, ANN, etc.
    """
    audio, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)  # shape: (40,)


# ==============================================================================
# FEATURE 2: MFCC Sequence
# Time-series version of MFCC for sequential models.
# Output: 2D array of shape (max_frames, n_mfcc) = (100, 40)
# ==============================================================================

def extract_mfcc_sequence(file_path, n_mfcc=40, max_frames=100):
    """
    Extract MFCC as a time-series sequence.
    The LSTM itself acts as the feature extractor over this sequence.
    Used by: LSTM model
    """
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc = mfcc.T  # shape: (frames, n_mfcc)

        if mfcc.shape[0] < max_frames:
            padding = np.zeros((max_frames - mfcc.shape[0], n_mfcc))
            mfcc = np.vstack((mfcc, padding))
        else:
            mfcc = mfcc[:max_frames, :]

        return mfcc  # shape: (max_frames, n_mfcc)
    except Exception as e:
        print(f"Error extracting MFCC sequence from {file_path}: {e}")
        return np.zeros((max_frames, n_mfcc))


# ==============================================================================
# FEATURE 3: Pitch (Fundamental Frequency - F0) Features
# Very voice-specific. Captures how the voice pitch changes over time.
# Deepfake/TTS voices often have unnatural pitch patterns (too flat or robotic).
# Output: 1D vector of shape (5,) — [mean, std, min, max, range of F0]
# ==============================================================================

def extract_pitch_features(file_path):
    """
    Extract statistical features from the fundamental frequency (F0/pitch).
    Deepfakes often have unnaturally flat or monotonic pitch contours.
    Used by: Classical ML models
    """
    try:
        audio, sr = librosa.load(file_path, sr=22050)

        # Extract pitch using YIN algorithm
        f0 = librosa.yin(audio, fmin=50, fmax=400)  # human voice range

        # Remove unvoiced frames (where pitch is 0 or very low)
        voiced_f0 = f0[f0 > 60]

        if len(voiced_f0) == 0:
            return np.zeros(5)

        features = np.array([
            np.mean(voiced_f0),    # average pitch
            np.std(voiced_f0),     # pitch variation (low = monotone/fake)
            np.min(voiced_f0),     # lowest pitch
            np.max(voiced_f0),     # highest pitch
            np.max(voiced_f0) - np.min(voiced_f0),  # pitch range (low = robotic)
        ])

        return features  # shape: (5,)
    except Exception as e:
        print(f"Error extracting pitch from {file_path}: {e}")
        return np.zeros(5)


# ==============================================================================
# FEATURE 4: Mel Filterbank Energies (Mel Spectrogram)
# Widely used in modern voice anti-spoofing (basis of many deep learning models).
# Represents the energy of speech in perceptually-scaled frequency bands.
# Output: 1D mean vector of shape (n_mels,) = (40,)
# ==============================================================================

def extract_mel_filterbank(file_path, n_mels=40):
    """
    Extract Mel Filterbank Energies.
    Closer to raw acoustic energy in speech than MFCC (no DCT applied).
    Preferred input for deep learning models. Widely used in ASVspoof research.
    Used by: CNN, LSTM, or as alternative to MFCC in classical models
    """
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)  # convert to log scale
        return np.mean(mel_db.T, axis=0)  # shape: (n_mels,)
    except Exception as e:
        print(f"Error extracting Mel Filterbank from {file_path}: {e}")
        return np.zeros(n_mels)


# ==============================================================================
# COMBINED VOICE FEATURE VECTOR
# Concatenates MFCC + Pitch + Mel Filterbank into one rich feature vector.
# Output shape: (40 + 5 + 40,) = (85,)
# ==============================================================================

def extract_combined_voice_features(file_path):
    """
    Extracts and concatenates MFCC, Pitch statistics, and Mel Filterbank energies.
    All three are voice-specific and complement each other:
      - MFCC:   captures vocal tract shape (what is being said)
      - Pitch:  captures vocal fold behavior (how it is said — naturalness)
      - Mel FB: captures raw spectral energy in speech bands
    Used by: Classical ML models, as a richer alternative to MFCC alone.
    """
    try:
        mfcc = extract_mfcc(file_path)              # (40,)
        pitch = extract_pitch_features(file_path)   # (5,)
        mel = extract_mel_filterbank(file_path)      # (40,)

        return np.concatenate([mfcc, pitch, mel])   # (85,)
    except Exception as e:
        print(f"Error extracting combined voice features from {file_path}: {e}")
        return np.zeros(85)