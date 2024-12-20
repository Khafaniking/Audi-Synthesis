#GANMODEL.py

import sys
print("Python version:", sys.version)

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers, Sequential, Model
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from PIL import Image
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.python.keras.layers import Dense, LeakyReLU, Layer, Conv2D, Conv2DTranspose, UpSampling2D, Activation, Reshape, Flatten
from tensorflow.image import ResizeMethod, resize
from keras.optimizers import Adam
from tensorflow.python.keras.layers import Input
from keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Dropout
from sklearn.metrics import mean_squared_error


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Directory paths for spectrograms and generated audio
real_spectrograms_dir = r'C:\Users\khafa\Downloads\UrbanSound8K\UrbanSound8K\street_music_samples\spectrograms'
output_audio_dir = r'C:\Users\khafa\Downloads\UrbanSound8K\UrbanSound8K\generated_audio'
genspec_dir = os.path.join(output_audio_dir, 'genspecs')
os.makedirs(output_audio_dir, exist_ok=True)
os.makedirs(genspec_dir, exist_ok=True)

# Model parameters
latent_dim = 128  # Latent dimension for generator input
spectrogram_shape = (256, 256, 1)  # Example shape, modify to match your data
sample_rate = 22050

def load_spectrograms(target_shape):
    spectrograms = []
    for filename in os.listdir(real_spectrograms_dir):
        if filename.endswith('.png'):
            img = plt.imread(os.path.join(real_spectrograms_dir, filename))  # Load image
            if len(img.shape) == 2:  # If image is 2D (grayscale without a channel)
                img = np.expand_dims(img, axis=-1)  # Add a channel dimension
            elif len(img.shape) == 3 and img.shape[-1] > 1:  # If multi-channel (e.g., RGB)
                img = img[..., :1]  # Keep only the first channel
            img_resized = tf.image.resize(img, target_shape[:2])  # Resize to target shape
            spectrograms.append(img_resized)
    # Convert to NumPy array and normalize to [-1, 1] for tanh
    spectrograms = np.array(spectrograms)
    spectrograms = spectrograms / 127.5 - 1.0  # Rescale pixel values [0, 255] -> [-1, 1]
    return spectrograms



real_spectrograms = load_spectrograms(spectrogram_shape)

# Inspect loaded spectrograms
print(f"Loaded spectrogram shape: {real_spectrograms.shape}")
plt.imshow(real_spectrograms[0].squeeze(), cmap='gray')
plt.title("Loaded Spectrogram (First Example)")
plt.colorbar()
plt.show()

#original

# Reduce the dataset size for experimentation/expediency
target_size = 100  # Number of samples to keep, we can adjust as needed
np.random.seed(42)
selected_indices = np.random.choice(len(real_spectrograms), size=target_size, replace=False)
real_spectrograms = real_spectrograms[selected_indices]

# Inspect reduced dataset just to confirm
print(f"Reduced dataset shape: {real_spectrograms.shape}")
plt.imshow(real_spectrograms[0].squeeze(), cmap='gray')
plt.title("Reduced Real Spectrogram (First Example)")
plt.colorbar()
plt.show()

#borrowed from SpecGAN https://github.com/chrisdonahue/wavegan/blob/master/specgan.py
#Copyright (c) 2019 Christopher Donahue
class CustomConv2DTranspose(Layer):
    def __init__(self, filters, kernel_size, strides=2, padding='same', upsample='zeros', **kwargs):
        super(CustomConv2DTranspose, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.upsample = upsample

        if self.upsample == 'zeros':
            self.conv_transpose = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)
        else:
            self.conv = Conv2D(filters, kernel_size, strides=1, padding=padding)
    
    def call(self, inputs):
        if self.upsample == 'zeros':
            return self.conv_transpose(inputs)
        else:
            if self.upsample == 'nn':
                method = ResizeMethod.NEAREST_NEIGHBOR
            elif self.upsample == 'linear':
                method = ResizeMethod.BILINEAR
            elif self.upsample == 'cubic':
                method = ResizeMethod.BICUBIC
            else:
                raise ValueError(f"Unsupported upsampling method: {self.upsample}")
            
            # Upsample the input
            upsampled = resize(inputs, [inputs.shape[1] * self.strides, inputs.shape[2] * self.strides], method=method)
            # Apply convolution
            return self.conv(upsampled)
    
    def get_config(self):
        config = super(CustomConv2DTranspose, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'upsample': self.upsample
        })
        return config


# Generator Model, supposed to follow a sort of drill down kind of approach
def generator(latent_dim, output_shape):
    model = Sequential([
        # Input layer: Dense + Reshape
        layers.Dense(16 * 16 * latent_dim * 8, activation='relu', input_dim=latent_dim),
        layers.Reshape((16, 16, latent_dim * 8)),

        # Layer 1
        CustomConv2DTranspose(latent_dim * 4, kernel_size=7, strides=2, upsample='linear'), 
        layers.BatchNormalization(),
        layers.ReLU(),

        # Layer 2
        CustomConv2DTranspose(latent_dim * 2, kernel_size=5, strides=2, upsample='linear'),
        layers.BatchNormalization(),
        layers.ReLU(),

        # Layer 3
        CustomConv2DTranspose(latent_dim, kernel_size=5, strides=2, upsample='cubic'),
        layers.BatchNormalization(),
        layers.ReLU(),

        # Layer 4 (ensure shape is 256 by 256)
        CustomConv2DTranspose(latent_dim // 2, kernel_size=3, strides=2, upsample='cubic'),
        layers.BatchNormalization(),
        layers.ReLU(),

        # Output Layer
        CustomConv2DTranspose(output_shape[-1], kernel_size=1, strides=1, upsample='nn'),
        layers.Activation('tanh'),
    ])
    return model

# Discriminator Model
def build_discriminator(input_shape):
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),

        layers.Conv2D(128, kernel_size=5, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),

        layers.Conv2D(256, kernel_size=3, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.3),

        layers.Conv2D(512, kernel_size=1, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.Flatten(),

        layers.Dense(1, activation='sigmoid')
    ])
    return model


# Instantiate models with the updated generator
generator = generator(latent_dim, spectrogram_shape)
discriminator = build_discriminator(spectrogram_shape)

# Optimizers and loss
generator_optimizer = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.95)
discriminator_optimizer = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.95)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# Custom training step with additional supervised loss for exact matching
@tf.function
def train_step(real_spectrograms, generator_updates=2, lambda_gp=25.0):
    batch_size = real_spectrograms.shape[0]
    noise = tf.random.normal([batch_size, latent_dim])
    
    # Train discriminator
    with tf.GradientTape() as disc_tape:
        real_spectrograms_noisy = real_spectrograms + tf.random.normal(real_spectrograms.shape, mean=0.0, stddev=0.02)
        generated_spectrograms = generator(noise, training=True)
        generated_spectrograms_noisy = generated_spectrograms #+ tf.random.normal(generated_spectrograms.shape, mean=0.0, stddev=0.01)
        #no additional noise need for the generated_spectrograms, might be doubling up
        #and harming the model
        real_output = discriminator(real_spectrograms_noisy, training=True)
        fake_output = discriminator(generated_spectrograms_noisy, training=True)

        real_labels = tf.ones_like(real_output) * 0.9  # Smoothing
        fake_labels = tf.zeros_like(fake_output)

        real_loss = cross_entropy(real_labels, real_output)
        fake_loss = cross_entropy(fake_labels, fake_output)

        # Dynamic weighting
        real_weight = 1.0 - tf.reduce_mean(real_output)
        fake_weight = tf.reduce_mean(fake_output)

        disc_loss = real_weight * real_loss + fake_weight * fake_loss

        # Gradient penalty computation
        alpha = tf.random.uniform([batch_size, 1, 1, 1], minval=0., maxval=1.)
        mixed_images = alpha * real_spectrograms + (1 - alpha) * generated_spectrograms
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(mixed_images)
            mixed_output = discriminator(mixed_images, training=True)
        gradients = gp_tape.gradient(mixed_output, [mixed_images])[0]
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(tf.maximum(gradients_norm - 1.0, 0)))

        # Add gradient penalty to discriminator loss
        disc_loss += lambda_gp * gradient_penalty

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    clipped_gradients = [tf.clip_by_value(grad, -0.5, 0.5) for grad in gradients_of_discriminator]
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Train generator multiple times per discriminator update
    for _ in range(generator_updates):
        with tf.GradientTape() as gen_tape:
            noise = tf.random.normal([batch_size, latent_dim])
            generated_spectrograms = generator(noise, training=True)
            fake_output = discriminator(generated_spectrograms, training=True)

            adversarial_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

            gen_loss = adversarial_loss

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return disc_loss, gen_loss, tf.reduce_mean(real_output), tf.reduce_mean(fake_output)

def compute_rmse(real_batch, generated_batch, target_shape=None):
    """
    Computes the average RMSE across a batch of real and generated spectrograms.
    Optionally resizes the spectrograms to the target shape.
    """
    rmse_values = []
    for real, gen in zip(real_batch, generated_batch):
        if target_shape:
            real = tf.image.resize(real, target_shape[:2]).numpy()
            gen = tf.image.resize(gen, target_shape[:2]).numpy()
        
        rmse = np.sqrt(mean_squared_error(real.flatten(), gen.flatten()))
        rmse_values.append(rmse)
    
    return np.mean(rmse_values)


def train_gan(real_spectrograms, epochs=1000, batch_size=8, generator_updates=2):
    rmse_values = [] 

    for epoch in range(epochs):
        # Shuffle dataset indices for each epoch because I'm paranoid
        shuffled_indices = np.random.permutation(len(real_spectrograms))
        real_spectrograms = real_spectrograms[shuffled_indices]

        epoch_rmse = 0

        # Iterate through mini-batches
        for i in range(real_spectrograms.shape[0] // batch_size):
            real_batch = real_spectrograms[i * batch_size: (i + 1) * batch_size]

            
            d_loss, g_loss, real_confidence, fake_confidence = train_step(real_batch, generator_updates)

            # Generate spectrograms for RMSE computation
            noise = tf.random.normal([real_batch.shape[0], latent_dim])
            generated_batch = generator.predict(noise)

            # Compute RMSE 
            batch_rmse = compute_rmse(real_batch, generated_batch, target_shape=(256, 256))
            epoch_rmse += batch_rmse

        # Average RMSE across all batches
        epoch_rmse /= (real_spectrograms.shape[0] // batch_size)
        rmse_values.append(epoch_rmse)

        # Visualize and save generated spectrograms for specific epochs
        if epoch == 0 or (epoch + 1) % 10 == 0:
            noise = tf.random.normal([batch_size, latent_dim])
            generated_spectrogram = generator.predict(noise)[0]  # Generate one spectrogram

            # Rescale from [-1, 1] to [0, 1]
            generated_spectrogram_rescaled = (generated_spectrogram + 1.0) / 2.0

            # Convert to power spectrogram (intensity approximation)
            power_spectrogram = generated_spectrogram_rescaled ** 2

            # Convert to dB scale for visualization
            spectrogram_db = librosa.power_to_db(power_spectrogram.squeeze(), ref=np.max)

            # Save the spectrogram as an image
            epoch_dir = os.path.join(genspec_dir, f"epoch_{epoch + 1}")
            os.makedirs(epoch_dir, exist_ok=True)
            spectrogram_path = os.path.join(epoch_dir, f"generated_spectrogram_{epoch + 1}.png")

            plt.figure(figsize=(10, 6))
            librosa.display.specshow(
                spectrogram_db,
                sr=sample_rate,  
                hop_length=512,  
                x_axis="time",
                y_axis="mel",
            )
            plt.colorbar(format="%+2.0f dB", boundaries=np.linspace(-80, 0, 100))
            plt.title(f"Generated Spectrogram (Epoch {epoch + 1})")
            plt.savefig(spectrogram_path)
            plt.close()

        # Print epoch results
        print(f"Epoch {epoch + 1}, Discriminator Loss: {d_loss:.6f}, Generator Loss: {g_loss:.6f}, "
              f"Real Confidence: {real_confidence:.6f}, Fake Confidence: {fake_confidence:.6f}, "
              f"RMSE: {epoch_rmse:.6f}")

    # Save RMSE values to a file
    np.savetxt("rmse_values.txt", rmse_values)





# Convert generated spectrogram to audio
#def convert_spectrogram_to_audio(spectrogram):
    #spectrogram = np.squeeze(spectrogram)

    # Ensure spectrogram is finite
    #if not np.isfinite(spectrogram).all():
        #raise ValueError("Spectrogram contains non-finite values.")

    # Convert spectrogram to audio
    #S = librosa.db_to_power(spectrogram)
    #audio = librosa.feature.inverse.mel_to_audio(S, sr=sample_rate, n_fft=2048, hop_length=512)
    #return audio

# Generate and save multiple audio samples
#def generate_audio_samples_and_comparisons(num_samples=5):
    #generated_spectrograms = []
    #for i in range(num_samples):
        # Generate noise input for the generator
        #noise = np.random.normal(0, 1, (1, latent_dim))
        # Generate a spectrogram from the noise
        #generated_spectrogram = generator.predict(noise)[0]
        #generated_spectrograms.append(generated_spectrogram)

        # Convert the generated spectrogram to audio
        #audio = convert_spectrogram_to_audio(generated_spectrogram)
        #output_file = os.path.join(output_audio_dir, f"generated_audio_{i}.wav")
        #sf.write(output_file, audio, sample_rate)
        #print(f"Saved generated audio to {output_file}")

    # Optionally, return the generated spectrograms for inspection
    #return np.array(generated_spectrograms)



# Generate and save audio samples after training
#generate_audio_samples_and_comparisons(num_samples=5)

# Train the GAN
train_gan(real_spectrograms, epochs=1000, batch_size=8, generator_updates=2)
