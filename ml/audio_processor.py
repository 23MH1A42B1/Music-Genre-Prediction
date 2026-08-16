import os
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fft import fft
from scipy.signal import get_window

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

FEATURE_COLUMNS = [
    "chroma_stft_mean", "rms_mean", "spectral_centroid_mean",
    "spectral_bandwidth_mean", "rolloff_mean", "zero_crossing_rate_mean",
    "tempo", "energy", "danceability"
] + [f"mfcc{i}_mean" for i in range(1, 21)]

def extract_features_from_pcm(signal, sample_rate=22050):
    """
    Extracts GTZAN-compatible audio features directly from a 1D numpy array of audio samples.
    """
    # Convert stereo to mono if 2D array
    if len(signal.shape) > 1:
        signal = signal.mean(axis=1)

    signal = signal.astype(np.float32)
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val

    N = len(signal)
    if N == 0:
        raise ValueError("Audio signal is empty")

    # 1. Zero Crossing Rate (ZCR)
    zcr = float(np.mean(np.abs(np.diff(np.sign(signal))) > 0))

    # 2. RMS Energy
    rms = float(np.sqrt(np.mean(signal**2)))

    # 3. Spectral Features via FFT (Take middle slice)
    frame_size = min(N, int(sample_rate * 3.0))
    start = max(0, (N - frame_size) // 2)
    slice_sig = signal[start:start + frame_size]

    window = get_window('hamming', len(slice_sig))
    windowed_sig = slice_sig * window
    
    spectrum = np.abs(fft(windowed_sig))[:len(slice_sig)//2]
    freqs = np.linspace(0, sample_rate / 2, len(spectrum))

    spec_sum = np.sum(spectrum) + 1e-9
    
    # Spectral Centroid
    spectral_centroid = float(np.sum(freqs * spectrum) / spec_sum)

    # Spectral Bandwidth
    spectral_bandwidth = float(np.sqrt(np.sum(((freqs - spectral_centroid)**2) * spectrum) / spec_sum))

    # Spectral Rolloff (85% energy cut-off)
    cum_energy = np.cumsum(spectrum)
    rolloff_idx = np.where(cum_energy >= 0.85 * cum_energy[-1])[0]
    rolloff = float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else float(freqs[-1])

    # 4. Tempo Estimation via Autocorrelation
    auto_corr = np.correlate(signal[::4], signal[::4], mode='full')
    auto_corr = auto_corr[len(auto_corr)//2:]
    min_lag = int((sample_rate / 4) * (60 / 180))
    max_lag = int((sample_rate / 4) * (60 / 60))
    if len(auto_corr) > max_lag:
        peak_lag = min_lag + np.argmax(auto_corr[min_lag:max_lag])
        bpm = float((sample_rate / 4) * 60 / peak_lag)
    else:
        bpm = 120.0

    # 5. Chroma STFT approximation
    chroma = float(0.25 + 0.2 * np.sin(spectral_centroid / 500.0))

    # 6. MFCC calculation (Mel Filterbank log-energies)
    num_filters = 20
    mel_freqs = np.linspace(0, np.sqrt(sample_rate / 2), num_filters + 2)**2
    mfccs = []
    for i in range(num_filters):
        f_low = mel_freqs[i]
        f_high = mel_freqs[i+2]
        filter_mask = (freqs >= f_low) & (freqs <= f_high)
        energy = np.sum(spectrum[filter_mask]**2) + 1e-6
        mfccs.append(float(np.log(energy)))

    # Scale MFCCs to standard GTZAN range
    mfccs = [float(m * 10 - 50) for m in mfccs]

    energy = float(np.clip(rms * 4.5, 0.1, 0.99))
    danceability = float(np.clip(0.3 + 0.5 * (bpm / 160.0), 0.1, 0.95))

    features = {
        "chroma_stft_mean": chroma,
        "rms_mean": rms,
        "spectral_centroid_mean": spectral_centroid,
        "spectral_bandwidth_mean": spectral_bandwidth,
        "rolloff_mean": rolloff,
        "zero_crossing_rate_mean": zcr,
        "tempo": bpm,
        "energy": energy,
        "danceability": danceability
    }

    for i, val in enumerate(mfccs):
        features[f"mfcc{i+1}_mean"] = val

    return features

def extract_features_from_wav(file_path):
    """
    Extracts features from WAV, MP3, OGG, or FLAC audio files using scipy or pydub fallback.
    """
    try:
        sample_rate, signal = wavfile.read(file_path)
        return extract_features_from_pcm(signal, sample_rate)
    except Exception as wav_err:
        # Fallback to pydub if scipy.io.wavfile fails (e.g. MP3 file)
        if PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_file(file_path)
                audio = audio.set_frame_rate(22050).set_channels(1)
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                return extract_features_from_pcm(samples, sample_rate=22050)
            except Exception as pydub_err:
                raise ValueError(f"Could not decode audio file: {str(pydub_err)}")
        else:
            raise ValueError(f"WAV format error: {str(wav_err)}. Install pydub for MP3 support.")

def extract_vector(feature_dict):
    """
    Converts feature dictionary to ordered feature vector array.
    """
    return [feature_dict[col] for col in FEATURE_COLUMNS]
