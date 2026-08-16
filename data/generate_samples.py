import os
import numpy as np
import scipy.io.wavfile as wavfile
import pandas as pd

# Directory setup
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_SAMPLES_DIR = os.path.join(DATA_DIR, "audio_samples")
os.makedirs(AUDIO_SAMPLES_DIR, exist_ok=True)

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

SAMPLE_RATE = 22050  # Standard audio sample rate
DURATION = 5  # 5 seconds sample per track

def synthesize_genre_audio(genre, sample_rate=22050, duration=5):
    """
    Synthesizes characteristic audio waveforms for each genre to simulate real music tracks.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    if genre == "blues":
        # Slow 12-bar blues progression synth + dominant 7th harmonics
        base_freq = 146.83  # D3
        signal = 0.5 * np.sin(2 * np.pi * base_freq * t)
        signal += 0.3 * np.sin(2 * np.pi * base_freq * 1.25 * t)  # Major 3rd
        signal += 0.2 * np.sin(2 * np.pi * base_freq * 1.5 * t)   # 5th
        # Add subtle shuffle rhythm envelope
        lfo = 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t)
        signal *= lfo

    elif genre == "classical":
        # Piano/strings arpeggiated harmony with smooth ADSR modulation
        freqs = [261.63, 329.63, 392.00, 523.25]  # C Major triad + octave
        signal = np.zeros_like(t)
        note_len = 0.25
        for i, freq in enumerate(t):
            note_idx = int((i / sample_rate) / note_len) % len(freqs)
            f = freqs[note_idx]
            signal[i] = 0.4 * np.sin(2 * np.pi * f * freq) + 0.2 * np.sin(2 * np.pi * f * 2 * freq)

    elif genre == "country":
        # Acoustic guitar fingerpicking pattern with twang
        freqs = [196.00, 246.94, 293.66, 392.00]  # G major
        signal = np.zeros_like(t)
        pattern_len = 0.2
        for i, freq in enumerate(t):
            idx = int((i / sample_rate) / pattern_len) % len(freqs)
            f = freqs[idx]
            signal[i] = 0.5 * np.sin(2 * np.pi * f * freq) + 0.15 * np.sin(2 * np.pi * f * 3 * freq)

    elif genre == "disco":
        # 120 BPM 4-on-the-floor kick pulse + octave bassline
        kick_freq = 4 * (120 / 60)  # 2 Hz beat pulse
        kick = np.exp(-10 * (t % 0.5)) * np.sin(2 * np.pi * 60 * t)
        bass = 0.4 * np.sin(2 * np.pi * 110 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * kick_freq * 2 * t))
        hihat = 0.1 * (np.random.rand(len(t)) - 0.5) * (np.sin(2 * np.pi * 8 * t) > 0)
        signal = kick + bass + hihat

    elif genre == "hiphop":
        # 90 BPM heavy 808 sub bass + crisp snare & hi-hats
        sub_bass = 0.6 * np.sin(2 * np.pi * 45 * t + 0.3 * np.sin(2 * np.pi * 1.5 * t))
        beat = np.exp(-15 * (t % (60 / 90))) * np.sin(2 * np.pi * 50 * t)
        hats = 0.08 * (np.random.rand(len(t)) - 0.5) * (np.sin(2 * np.pi * 6 * t) > 0.5)
        signal = sub_bass + beat + hats

    elif genre == "jazz":
        # Swing tempo, major 7th chord swells, walking basslines
        f1, f2, f3, f4 = 174.61, 220.00, 261.63, 329.63  # F maj7
        swing = 0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t)
        signal = (0.3 * np.sin(2 * np.pi * f1 * t) +
                  0.2 * np.sin(2 * np.pi * f2 * t) +
                  0.2 * np.sin(2 * np.pi * f3 * t) +
                  0.2 * np.sin(2 * np.pi * f4 * t)) * swing

    elif genre == "metal":
        # Fast double kick (160 BPM), heavy distortion & high energy harmonics
        drive = 3.0
        raw_riff = np.sin(2 * np.pi * 82.41 * t) + 0.8 * np.sin(2 * np.pi * 123.47 * t)
        distorted = np.tanh(drive * raw_riff)  # Hard clipping distortion
        fast_kick = 0.4 * np.exp(-25 * (t % 0.1875)) * np.sin(2 * np.pi * 70 * t)
        signal = 0.7 * distorted + fast_kick

    elif genre == "pop":
        # Upbeat melodic synth chord progression (115 BPM)
        chords = [261.63, 349.23, 220.00, 392.00]  # C - F - Am - G
        chord_len = 1.0
        signal = np.zeros_like(t)
        for i, freq in enumerate(t):
            c_idx = int((i / sample_rate) / chord_len) % len(chords)
            f = chords[c_idx]
            signal[i] = 0.4 * np.sin(2 * np.pi * f * freq) + 0.3 * np.sin(2 * np.pi * f * 2 * freq)
        beat = 0.3 * np.exp(-12 * (t % 0.52)) * np.sin(2 * np.pi * 90 * t)
        signal += beat

    elif genre == "reggae":
        # Off-beat guitar skank (75 BPM) + heavy dub bassline
        offbeat = (np.sin(2 * np.pi * 2.5 * t) > 0.3).astype(float)
        skank = 0.4 * np.sin(2 * np.pi * 329.63 * t) * offbeat
        dub_bass = 0.6 * np.sin(2 * np.pi * 55 * t) * (0.8 + 0.2 * np.sin(2 * np.pi * 1.25 * t))
        signal = skank + dub_bass

    elif genre == "rock":
        # Power chords (E5, G5, A5) + driving drum beat (125 BPM)
        riff_freqs = [82.41, 98.00, 110.00]
        riff_len = 0.8
        signal = np.zeros_like(t)
        for i, freq in enumerate(t):
            r_idx = int((i / sample_rate) / riff_len) % len(riff_freqs)
            f = riff_freqs[r_idx]
            signal[i] = 0.5 * np.tanh(2.0 * np.sin(2 * np.pi * f * freq)) + 0.3 * np.sin(2 * np.pi * f * 1.5 * freq)
        drums = 0.35 * np.exp(-15 * (t % 0.48)) * np.sin(2 * np.pi * 80 * t)
        signal += drums

    else:
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Normalize audio to prevent clipping
    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    audio_int16 = (signal * 32767).astype(np.int16)
    return audio_int16

def create_sample_files_and_dataset(num_samples_per_genre=100):
    """
    Creates audio WAV sample files for each genre and generates a dataset DataFrame matching GTZAN features.
    """
    print("Generating audio sample WAV files...")
    for genre in GENRES:
        wav_path = os.path.join(AUDIO_SAMPLES_DIR, f"{genre}_sample.wav")
        audio_data = synthesize_genre_audio(genre, SAMPLE_RATE, DURATION)
        wavfile.write(wav_path, SAMPLE_RATE, audio_data)
        print(f"  Created: {wav_path}")

    print(f"Constructing GTZAN dataset benchmark with {num_samples_per_genre * len(GENRES)} rows...")
    
    np.random.seed(42)
    dataset_rows = []

    # Characteristic mean audio feature profiles per genre for statistical realism
    genre_profiles = {
        "blues":     {"chroma": 0.34, "rms": 0.11, "spec_cent": 1950, "spec_bw": 2050, "rolloff": 4100, "zcr": 0.085, "tempo": 105, "energy": 0.55},
        "classical": {"chroma": 0.27, "rms": 0.04, "spec_cent": 1350, "spec_bw": 1500, "rolloff": 2600, "zcr": 0.042, "tempo": 82,  "energy": 0.25},
        "country":   {"chroma": 0.35, "rms": 0.12, "spec_cent": 2100, "spec_bw": 2200, "rolloff": 4400, "zcr": 0.092, "tempo": 118, "energy": 0.58},
        "disco":     {"chroma": 0.40, "rms": 0.16, "spec_cent": 2650, "spec_bw": 2500, "rolloff": 5500, "zcr": 0.115, "tempo": 120, "energy": 0.78},
        "hiphop":    {"chroma": 0.38, "rms": 0.18, "spec_cent": 2250, "spec_bw": 2400, "rolloff": 4800, "zcr": 0.098, "tempo": 92,  "energy": 0.82},
        "jazz":      {"chroma": 0.32, "rms": 0.07, "spec_cent": 1750, "spec_bw": 1900, "rolloff": 3600, "zcr": 0.068, "tempo": 98,  "energy": 0.40},
        "metal":     {"chroma": 0.43, "rms": 0.22, "spec_cent": 3100, "spec_bw": 2700, "rolloff": 6300, "zcr": 0.142, "tempo": 145, "energy": 0.95},
        "pop":       {"chroma": 0.39, "rms": 0.17, "spec_cent": 2500, "spec_bw": 2450, "rolloff": 5200, "zcr": 0.108, "tempo": 116, "energy": 0.75},
        "reggae":    {"chroma": 0.36, "rms": 0.12, "spec_cent": 2000, "spec_bw": 2150, "rolloff": 4200, "zcr": 0.082, "tempo": 76,  "energy": 0.52},
        "rock":      {"chroma": 0.38, "rms": 0.15, "spec_cent": 2400, "spec_bw": 2350, "rolloff": 5000, "zcr": 0.102, "tempo": 128, "energy": 0.72},
    }

    for genre in GENRES:
        prof = genre_profiles[genre]
        for i in range(num_samples_per_genre):
            row = {"filename": f"{genre}.{i:05d}.wav", "label": genre}
            
            # Extract main features with realistic gaussian variance
            chroma = max(0.05, float(np.random.normal(prof["chroma"], 0.03)))
            rms = max(0.01, float(np.random.normal(prof["rms"], 0.02)))
            spec_cent = max(500, float(np.random.normal(prof["spec_cent"], 200)))
            spec_bw = max(500, float(np.random.normal(prof["spec_bw"], 180)))
            rolloff = max(1000, float(np.random.normal(prof["rolloff"], 350)))
            zcr = max(0.01, float(np.random.normal(prof["zcr"], 0.015)))
            tempo = max(50, float(np.random.normal(prof["tempo"], 8)))
            energy = float(np.clip(np.random.normal(prof["energy"], 0.08), 0.0, 1.0))
            danceability = float(np.clip(0.3 + 0.6 * (tempo / 160) + np.random.normal(0, 0.05), 0.1, 0.95))

            row.update({
                "chroma_stft_mean": chroma,
                "rms_mean": rms,
                "spectral_centroid_mean": spec_cent,
                "spectral_bandwidth_mean": spec_bw,
                "rolloff_mean": rolloff,
                "zero_crossing_rate_mean": zcr,
                "tempo": tempo,
                "energy": energy,
                "danceability": danceability
            })

            # Generate 20 MFCC coefficient means
            mfcc_base_weights = {
                "blues":     [-150, 120, -10, 30, -5, 15, -8, 10, -5, 8, -4, 6, -3, 5, -2, 4, -2, 3, -1, 2],
                "classical": [-250, 150, 20, -10, 15, -5, 10, -3, 8, -2, 6, -1, 5, 0, 4, 1, 3, 1, 2, 1],
                "country":   [-130, 110, -15, 35, -8, 18, -6, 12, -4, 9, -3, 7, -2, 6, -2, 5, -1, 4, -1, 3],
                "disco":     [-80,  90,  -25, 45, -12, 22, -10, 15, -8, 11, -5, 9, -4, 8, -3, 6, -2, 5, -2, 4],
                "hiphop":    [-60,  80,  -35, 50, -15, 25, -12, 18, -9, 13, -6, 10, -5, 9, -4, 7, -3, 6, -2, 5],
                "jazz":      [-180, 135,  10, 15,  5,  10,  2,  8,  0,  6, -1, 5, -1, 4, -1, 3,  0, 2,  0, 2],
                "metal":     [-40,  65,  -45, 60, -20, 30, -15, 22, -12, 16, -8, 12, -6, 11, -5, 9, -4, 8, -3, 6],
                "pop":       [-90,  95,  -20, 40, -10, 20, -8,  14, -6, 10, -4, 8, -3, 7, -2, 5, -2, 4, -1, 3],
                "reggae":    [-120, 105, -18, 32, -8,  16, -6,  11, -4,  8, -3, 6, -2, 5, -2, 4, -1, 3, -1, 2],
                "rock":      [-70,  85,  -30, 48, -14, 24, -11, 16, -8, 12, -5, 10, -4, 8, -3, 7, -2, 5, -2, 4],
            }
            mfccs = mfcc_base_weights[genre]
            for m in range(1, 21):
                row[f"mfcc{m}_mean"] = float(mfccs[m-1] + np.random.normal(0, 3.0))

            dataset_rows.append(row)

    df = pd.DataFrame(dataset_rows)
    csv_path = os.path.join(DATA_DIR, "gtzan_features.csv")
    df.to_csv(csv_path, index=False)
    print(f"Dataset successfully saved to: {csv_path}")
    return csv_path

if __name__ == "__main__":
    create_sample_files_and_dataset()
