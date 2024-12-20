import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.metrics import mean_squared_error

# Directory paths for spectrograms and generated audio
real_spectrograms_dir = r'C:\Users\khafa\Downloads\UrbanSound8K\UrbanSound8K\street_music_samples\spectrograms'
genspec_dir = r'C:\Users\khafa\Downloads\UrbanSound8K\UrbanSound8K\generated_audio\genspecs'

# Parameters
target_shape = (256, 256)  # Resize all spectrograms to this shape


def load_spectrogram(file_path):
    """
    Load spectrogram from an image file.
    """
    img = plt.imread(file_path)
    if img.ndim == 3:  # If RGB, convert to grayscale
        img = img[..., 0]
    return img


def resize_spectrogram(spectrogram, target_shape):
    """
    Resize spectrogram to the target shape using skimage.
    """
    return resize(spectrogram, target_shape, anti_aliasing=True)


def compute_rmse(real, generated, target_shape):
    """
    Compute RMSE between real and generated spectrograms after resizing.
    """
    # Resize generated spectrogram to match real spectrogram's shape
    generated_resized = resize_spectrogram(generated, target_shape)
    # Flatten both spectrograms for RMSE computation
    real_flatten = real.flatten()
    generated_flatten = generated_resized.flatten()
    return np.sqrt(mean_squared_error(real_flatten, generated_flatten))


def calculate_rmse(real_spectrograms_dir, genspec_dir, target_shape=(256, 256)):
    """
    Calculate RMSE between generated and real spectrograms for all epochs.
    """
    # Load all real spectrograms
    real_spectrograms = []
    for filename in os.listdir(real_spectrograms_dir):
        if filename.endswith('.png'):
            real_spectrogram = load_spectrogram(os.path.join(real_spectrograms_dir, filename))
            real_spectrograms.append(real_spectrogram)

    real_spectrograms = np.array(real_spectrograms)
    print(f"Loaded {len(real_spectrograms)} real spectrograms for comparison.")

    rmse_values = []

    # Iterate over generated spectrograms in each epoch folder
    for epoch_folder in os.listdir(genspec_dir):
        epoch_path = os.path.join(genspec_dir, epoch_folder)
        if not os.path.isdir(epoch_path):
            continue

        for filename in os.listdir(epoch_path):
            if filename.endswith('.png'):
                gen_spec_path = os.path.join(epoch_path, filename)
                generated_spectrogram = load_spectrogram(gen_spec_path)

                # Compare against all real spectrograms
                rmses_for_spec = [
                    compute_rmse(real_spec, generated_spectrogram, target_shape)
                    for real_spec in real_spectrograms
                ]
                average_rmse = np.mean(rmses_for_spec)  # Average RMSE over all comparisons
                rmse_values.append((epoch_folder, filename, average_rmse))

    # Save RMSE values to a file
    output_file = os.path.join(genspec_dir, 'rmse_values.txt')
    with open(output_file, 'w') as f:
        for epoch_folder, filename, rmse in rmse_values:
            f.write(f"Epoch: {epoch_folder}, File: {filename}, RMSE: {rmse:.6f}\n")
    print(f"RMSE values saved to {output_file}")


# Run the RMSE calculation
calculate_rmse(real_spectrograms_dir, genspec_dir, target_shape)
