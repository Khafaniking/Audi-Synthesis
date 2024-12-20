import os
import numpy as np
from scipy.linalg import sqrtm
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.image import resize
import matplotlib.pyplot as plt

# Paths to directories
real_spectrograms_dir = r'C:\Users\khafa\Downloads\UrbanSound8K\UrbanSound8K\street_music_samples\spectrograms'
genspec_dir = r'C:\Users\khafa\Downloads\UrbanSound8K\UrbanSound8K\generated_audio\genspecs'

# Load InceptionV3 model (without the top layer, for feature extraction)
inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(256, 256, 3))


def preprocess_and_load_images(directory, target_size=(256, 256)):
    """Load and preprocess images from a directory."""
    images = []
    for root, _, files in os.walk(directory):  # Recursively walk through all subfolders
        for filename in files:
            if filename.endswith('.png'):
                filepath = os.path.join(root, filename)
                img = plt.imread(filepath)

                # Ensure all images have 3 channels (convert RGBA to RGB)
                if img.shape[-1] == 4:  # RGBA
                    img = img[..., :3]  # Drop the alpha channel
                elif len(img.shape) == 2:  # Grayscale
                    img = np.stack([img] * 3, axis=-1)  # Convert to RGB

                img_resized = resize(img, target_size).numpy()
                images.append(img_resized)
    images = np.array(images)
    images = preprocess_input(images)  # Normalize for InceptionV3
    return images


def calculate_fid(real_images, generated_images):
    """Compute the Frechet Inception Distance (FID)."""
    real_features = inception_model.predict(real_images, verbose=0)
    generated_features = inception_model.predict(generated_images, verbose=0)

    # Calculate mean and covariance
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)

    # Add small noise to avoid singular covariance matrices
    sigma_real += np.eye(sigma_real.shape[0]) * 1e-6
    sigma_gen += np.eye(sigma_gen.shape[0]) * 1e-6

    # Calculate FID
    mean_diff = np.sum((mu_real - mu_gen) ** 2)
    covmean, _ = sqrtm(sigma_real @ sigma_gen, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = mean_diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid


# Load real spectrograms
real_images = preprocess_and_load_images(real_spectrograms_dir)
print(f"Total real images loaded: {len(real_images)}")

# Subsample real images to match the training subset size (e.g., 50)
np.random.seed(42)  # Ensure reproducibility
real_images_subset = real_images[np.random.choice(len(real_images), size=1000, replace=True)]
print(f"Number of subsampled real images: {len(real_images_subset)}")

# Process each epoch folder in genspec_dir
fid_scores = {}
all_generated_images = []
for epoch_folder in sorted(os.listdir(genspec_dir)):
    epoch_path = os.path.join(genspec_dir, epoch_folder)
    if os.path.isdir(epoch_path):
        print(f"Processing {epoch_folder}...")

        # Load the single generated image for the epoch
        generated_images = preprocess_and_load_images(epoch_path)

        if len(generated_images) == 0:
            print(f"No generated images in {epoch_folder}, skipping.")
            continue

        if len(generated_images) < 1000:
            # Repeat and truncate generated images to reach 1000
            generated_images = np.tile(generated_images, (1000 // len(generated_images) + 1, 1, 1, 1))[:1000]
        else:
            # Randomly sample 1000 generated images if there are more
            generated_images = generated_images[np.random.choice(len(generated_images), size=1000, replace=False)]

        # Calculate FID
        fid_score = calculate_fid(real_images_subset, generated_images)
        fid_scores[epoch_folder] = fid_score
        print(f"FID for {epoch_folder}: {fid_score:.4f}")

# Save FID scores to a file
fid_output_file = os.path.join(genspec_dir, 'fid_scores.txt')
with open(fid_output_file, 'w') as f:
    for epoch, score in fid_scores.items():
        f.write(f"{epoch}: {score:.4f}\n")

print(f"FID scores saved to {fid_output_file}")
