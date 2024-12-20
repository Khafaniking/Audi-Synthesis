import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf  # For resizing
import soundfile as sf

#partial adapted from https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
# Path
audio_dir = r'C:\Users\khafa\Downloads\UrbanSound8K\UrbanSound8K\street_music_samples'
spectrogram_dir = os.path.join(audio_dir, 'spectrograms')
os.makedirs(spectrogram_dir, exist_ok=True)

# Clear old spectrograms
for file in os.listdir(spectrogram_dir):
    file_path = os.path.join(spectrogram_dir, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

print(f"Cleared old spectrograms in {spectrogram_dir}")


# Spectrogram settings
target_sample_rate = 22050
n_mels = 64
hop_length = 512
n_fft = 1024
max_duration_ms = 4000  # 4 seconds
time_shift_limit = 0.4  # Percentage of sample length, totals to 5.6
resize_shape = (256, 256)  # Target resize shape (height, width)

# Function to rechannel audio
def rechannel_audio(y, sr, target_channels=2):
    if len(y.shape) == 1 and target_channels == 2:
        # Mono to stereo by duplicating the single channel
        y = np.stack([y, y])
    elif len(y.shape) == 2 and y.shape[0] == 2 and target_channels == 1:
        # Stereo to mono by selecting the first channel
        y = y[0]
    return y

# Function to resample audio
def resample_audio(y, orig_sr, target_sr):
    if orig_sr != target_sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    return y

# Function to pad or truncate audio
def pad_truncate_audio(y, sr, max_duration_ms):
    max_len = int(sr * (max_duration_ms / 1000))  # Maximum length in samples
    sig_len = y.shape[0]

    if sig_len > max_len:
        # Truncate audio
        y = y[:max_len]
    elif sig_len < max_len:
        # Pad audio
        pad_begin_len = random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len
        y = np.pad(y, (pad_begin_len, pad_end_len), mode='constant')
    return y

# Function to perform time shift
def time_shift_audio(y, shift_limit):
    shift_amt = int(random.random() * shift_limit * len(y))
    return np.roll(y, shift_amt)

# Process our audio files
for file_name in os.listdir(audio_dir):
    if file_name.endswith('.wav'):
        audio_path = os.path.join(audio_dir, file_name)

        # Load audio
        y, sr = librosa.load(audio_path, sr=None, mono=False)

        # Ensure audio is two channels
        y = rechannel_audio(y, sr, target_channels=2)

        # Resample audio to target sample rate
        y_resampled = resample_audio(y=np.mean(y, axis=0), orig_sr=sr, target_sr=target_sample_rate)

        # Pad or truncate audio to maximum duration
        y_padded_truncated = pad_truncate_audio(y_resampled, target_sample_rate, max_duration_ms)

        # Apply time shift
        y_time_shifted = time_shift_audio(y_padded_truncated, time_shift_limit)

        # Convert to Mel spectrogram
        S = librosa.feature.melspectrogram(y=np.mean(y, axis=0), sr=sr, n_mels=n_mels, hop_length=hop_length,
                                           n_fft=n_fft)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Resize the spectrogram
        S_dB_resized = tf.image.resize(S_dB[..., np.newaxis], resize_shape, method='bilinear').numpy().squeeze()
        print(f"Resized spectrogram shape for {file_name}: {S_dB_resized.shape}")

        # Save spectrogram
        fig, ax = plt.subplots()
        librosa.display.specshow(S_dB_resized, sr=target_sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax)
        plt.axis('off')
        spectrogram_path = os.path.join(spectrogram_dir, f"{file_name.replace('.wav', '')}_spectrogram.png")
        # Save resized spectrograms as grayscale images
        S_dB_resized = tf.image.resize(S_dB[..., np.newaxis], resize_shape, method='bilinear').numpy()
        tf.keras.utils.save_img(spectrogram_path, S_dB_resized, scale=True)

        plt.close(fig)

        print(f"Generated and resized spectrogram for {file_name} at {spectrogram_path}")

print(f"Finished generating resized spectrograms in {spectrogram_dir}")


def inspect_spectrogram_shapes(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            path = os.path.join(directory, filename)
            img = plt.imread(path)  # Read the spectrogram image
            print(f"{filename}: {img.shape}")

# Call the function to inspect the shapes
inspect_spectrogram_shapes(spectrogram_dir)
