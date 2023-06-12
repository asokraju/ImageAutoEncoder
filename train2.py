import os
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from tensorflow import shape as tf_shape
from tensorflow import exp as tf_exp
from tensorflow import square as tf_square
from tensorflow import reduce_sum, reduce_mean
from tensorflow import GradientTape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Layer, Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.backend import random_normal
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


# Sampling layer
class Sampling(Layer):
    "used to sample a vector in latent space with learned mean - z_mean and (log) variance - z_log_var"
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = tf_shape(z_mean)[0]
        vec_len = tf_shape(z_mean)[1]
        epsilon = random_normal(shape=(batch_size, vec_len))
        return z_mean + tf_exp(0.5 * z_log_var) * epsilon


# Define the encoder model
def encoder_model(input_shape, filters, dense_layer_dim, latent_dim):
    """
    Creates an encoder model for grayscale images that maps input images to a lower-dimensional latent space.
    
    Args:
    - input_shape: Tuple representing the shape of the input images (height, width, channels).
    - filters: List of integers representing the number of filters in each convolutional layer.
    - dense_layer_dim: Integer representing the number of neurons in the dense layer.
    - latent_dim: Integer representing the dimensionality of the latent space.
    
    Returns:
    - encoder: Keras Model object representing the encoder model.
    - encoder_layers_dim: List of tuples representing the dimensionality of each layer in the encoder.
    """
    # Create input layer
    encoder_layers_dim = []  # List to store the dimensions of each layer in the encoder
    
    # Define the input layer
    encoder_inputs = Input(shape=input_shape)
    encoder_layers_dim.append(tuple(encoder_inputs.shape[1:]))  # Add input layer dimensions to list
    
    # Add convolutional layers with specified number of filters and activation function
    x = Conv2D(filters[0], (3,3), activation="relu", strides=2, padding="same")(encoder_inputs)
    encoder_layers_dim.append(tuple(x.shape[1:]))  # Add conv layer dimensions to list
    
    # Add additional convolutional layers with specified number of filters and activation function
    mid_layers = [Conv2D(f, 3, activation="relu", strides=2, padding="same") for f in filters[1:]]
    for mid_layer in mid_layers:
        x = mid_layer(x)
        encoder_layers_dim.append(tuple(x.shape[1:]))  # Add mid layer dimensions to list
    
    # Flatten convolutional output to prepare for dense layers
    x = Flatten()(x)
    encoder_layers_dim.append(tuple(x.shape[1:]))  # Add flattened layer dimensions to list
    
    # Add dense layer with specified number of neurons and activation function
    x = Dense(dense_layer_dim, activation='relu')(x)
    
    # Add output layers for latent space (mean and variance) and sample from this space
    z_mean = Dense(latent_dim, name = "z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder_layers_dim.append(tuple(z.shape[1:]))  # Add output layer dimensions to list
    
    # Create encoder model
    return Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder'), encoder_layers_dim

# decoder model for grayscale images
def decoder_model(encoder_layers_dim):
    # Extract necessary dimensions from encoder model output
    latent_dim = encoder_layers_dim[-1][0]
    dense_layer_dim = encoder_layers_dim[-2][0]
    first_conv_layer_dim = encoder_layers_dim[-3]
    output_layer = encoder_layers_dim[0]

    # Create input layer for latent space vector
    latent_inputs = Input(shape=(latent_dim,))

    # Determine number of filters for each transpose convolutional layer
    filters = [f[-1] for f in encoder_layers_dim[1:-2]]

    # Feed latent vector through a dense layer with ReLU activation
    # Note that we apply the first filter in the form of dense and reshape it
    x = Dense(dense_layer_dim, activation="relu")(latent_inputs)
    x = Reshape(first_conv_layer_dim)(x)

    # Apply series of transpose convolutional layers with ReLU activation and same padding and Upsampling
    mid_layers = [Conv2DTranspose(f, 3, activation="relu", strides=2, padding="same") for f in filters[::-1]]
    for mid_layer in mid_layers:
        x = mid_layer(x)

    # Apply final convolutional layer with sigmoid activation to output reconstructed image
    decoder_outputs = Conv2DTranspose(output_layer[-1], 3, activation="sigmoid", padding="same")(x)

    # Create and return Keras model with latent vector as input and reconstructed image as output
    return Model(latent_inputs, decoder_outputs, name="decoder")


# VAE for GrayScale Images
class VAE(Model):
    """
    This is a Variational Autoencoder (VAE) implemented using the Keras Model API. 
    It has an encoder and a decoder network defined separately and passed to the constructor as arguments. 
    The VAE class inherits from the Keras Model class and overrides the train_step() method to define the training loop.

    During forward pass, the encoder takes an input image and outputs the mean and standard deviation 
    of a latent space distribution, as well as a sampled vector from that distribution. 
    The decoder takes the sampled vector and outputs a reconstructed image.

    The training loop consists of computing the reconstruction loss and the 
    KL divergence loss, and then computing gradients and updating weights using the Adam optimizer. 
    The reconstruction loss measures the difference between the input image and the reconstructed image,
    while the KL divergence loss measures the divergence between the latent space distribution and a standard normal distribution. 
    The total loss is the sum of the two losses.

    The VAE class also defines three metrics to track during training: the total loss, the reconstruction loss, 
    and the KL divergence loss. These metrics are updated in the train_step() method and can be accessed via the metrics property. 
    The train_step() method returns a dictionary of these metrics.
    
    """
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
        # Define metrics to track during training
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    # Define forward pass
    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, z, reconstruction

    # Define training step
    def train_step(self, data):
        with GradientTape() as tape:
            # Forward pass through encoder and decoder
            z_mean, z_log_var, z, reconstruction = self(data)
            
            # Compute reconstruction loss
            reconstruction_loss = reduce_mean(
                reduce_sum(
                    binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )

            # Compute KL divergence loss
            kl_loss = -0.5 * (1 + z_log_var - tf_square(z_mean) - tf_exp(z_log_var))
            kl_loss = reduce_mean(reduce_sum(kl_loss, axis=1))

            # Compute total loss
            total_loss = reconstruction_loss + kl_loss
            
        # Compute gradients and update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        # Return metrics as dictionary
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-image-shape', type=int, default=56)
    parser.add_argument('--filters', type=int, nargs='+', default=[32, 64])
    parser.add_argument('--dense-layer-dim', type=int, default=16)
    parser.add_argument('--latent-dim', type=int, default=6)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--train-split', type=float, default=0.8)
    
    args = parser.parse_args()
    return args


def get_image_data(all_dirs):
    # List to store all image file paths
    all_image_paths = []

    # Loop through all directories and subdirectories in the data directory
    for data_dir in all_dirs:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                # Check if the file is an image file (you can add more extensions as needed)
                if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                    # If the file is an image file, append its path to the list
                    all_image_paths.append(os.path.join(root, file))
        print(data_dir)
    image_count = len(all_image_paths)
    print("Total number of imges:", image_count)
    return all_image_paths

class VAECallback(Callback):
    """
    Randomly sample 5 images from validation_data set and shows the reconstruction after each epoch
    """
    def __init__(self, vae, validation_data, log_dir, n=5):
        self.vae = vae
        self.validation_data = validation_data
        self.n = n
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        # Generate decoded images from the validation input
        validation_batch = next(iter(self.validation_data))
        _, _, _, reconstructed_images = self.vae.predict(validation_batch)

        # Rescale pixel values to [0, 1]
        reconstructed_images = np.clip(reconstructed_images, 0.0, 1.0)

        # Plot the original and reconstructed images side by side
        plt.figure(figsize=(10, 2*self.n))  # Adjusted the figure size
        for i in range(self.n):
            plt.subplot(self.n, 2, 2*i+1)
            plt.imshow(validation_batch[i], cmap='gray')
            plt.axis('off')
            plt.subplot(self.n, 2, 2*i+2)
            plt.imshow(reconstructed_images[i], cmap='gray')
            plt.axis('off')
        plt.savefig(self.log_dir + '\\decoded_images_epoch_{:04d}.png'.format(epoch))
        # plt.show()


if __name__ == '__main__':
    
    args = parse_arguments()
    all_image_paths = get_image_data([r'C:\Users\kkosara\ImageAutoEncoder\Data'])
    image_count = len(all_image_paths)
    TRAIN_SPLIT  = args.train_split
    OUTPUT_IMAGE_SHAPE = args.output_image_shape
    INPUT_SHAPE = (OUTPUT_IMAGE_SHAPE, OUTPUT_IMAGE_SHAPE, 1)
    FILTERS = args.filters
    DENSE_LAYER_DIM = args.dense_layer_dim
    LATENT_DIM = args.latent_dim
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate

    LOGDIR = os.path.join(r"C:\Users\kkosara\ImageAutoEncoder\logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.mkdir(LOGDIR)

    df_train = pd.DataFrame({'image_paths': all_image_paths[:int(image_count*TRAIN_SPLIT)]})
    df_test = pd.DataFrame({'image_paths': all_image_paths[:int(image_count*(1-TRAIN_SPLIT))]})

    train_datagen_args = dict(
        rescale=1.0 / 255,  # Normalize pixel values between 0 and 1
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    test_datagen_args = dict(rescale=1.0 / 255)

    train_datagen = ImageDataGenerator(**train_datagen_args)
    test_datagen = ImageDataGenerator(**test_datagen_args)
    # Use flow_from_dataframe to generate data batches
    train_data_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        color_mode='grayscale',
        x_col='image_paths',
        y_col=None,
        target_size=(OUTPUT_IMAGE_SHAPE, OUTPUT_IMAGE_SHAPE),  # Specify the desired size of the input images
        batch_size=BATCH_SIZE,
        class_mode=None,  # Set to None since there are no labels
        shuffle=True  # Set to True for randomizing the order of the images
    )

    test_data_generator = test_datagen.flow_from_dataframe(
        dataframe=df_test,
        color_mode='grayscale',
        x_col='image_paths',
        y_col=None,
        target_size=(OUTPUT_IMAGE_SHAPE, OUTPUT_IMAGE_SHAPE),  # Specify the desired size of the input images
        batch_size=BATCH_SIZE,
        class_mode=None,  # Set to None since there are no labels
        shuffle=True  # Set to True for randomizing the order of the images
    )



    encoder, encoder_layers_dim = encoder_model(input_shape = INPUT_SHAPE, filters=FILTERS, dense_layer_dim=DENSE_LAYER_DIM, latent_dim=LATENT_DIM)
    print(encoder.summary())
    print(encoder_layers_dim)
    decoder = decoder_model(encoder_layers_dim)
    print(decoder.summary())
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=Adam(learning_rate=LEARNING_RATE))
    vae_callback = VAECallback(vae, test_data_generator, LOGDIR)
    tensorboard_cb = TensorBoard(log_dir=LOGDIR, histogram_freq=1)
    history = vae.fit(
        train_data_generator,
        epochs=EPOCHS,
        validation_data=test_data_generator,
        callbacks=[vae_callback]
    )

    encoder.save(LOGDIR +"/encoder", overwrite=True, save_format=None)
    decoder.save(LOGDIR +"/decoder", overwrite=True, save_format=None)
    vae.save(LOGDIR +"/vae", overwrite=True, save_format=None)