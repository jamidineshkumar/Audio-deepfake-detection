import librosa
import numpy as np


def extract_mfcc(file_path, n_mfcc=40):
    """
    Extract MFCC features from audio file (Mean version for Classical ML)
    """
    audio, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # take mean across time axis
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

def extract_mfcc_sequence(file_path, n_mfcc=40, max_frames=100):
    """
    Extract MFCC sequence features for LSTM (Time-series version)
    """
    try:
        audio, sr = librosa.load(file_path, sr=22050, duration=max_frames * 0.02) # approx duration
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc = mfcc.T # (frames, n_mfcc)

        # Pad or truncate to max_frames
        if mfcc.shape[0] < max_frames:
            padding = np.zeros((max_frames - mfcc.shape[0], n_mfcc))
            mfcc = np.vstack((mfcc, padding))
        else:
            mfcc = mfcc[:max_frames, :]
            
        return mfcc
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return np.zeros((max_frames, n_mfcc))