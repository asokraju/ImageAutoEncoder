
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as  np


# Sampling layer
class Sampling(layers.Layer):
    "used to sample a vector in latent space with learned mean - z_mean and (log) variance - z_log_var"
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        vec_len = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch_size, vec_len))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Define the encoder model
def encoder_model_gs(input_shape, filters, dense_layer_dim, latent_dim):
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
    encoder_inputs = keras.Input(shape=input_shape)
    encoder_layers_dim.append(tuple(encoder_inputs.shape[1:]))  # Add input layer dimensions to list
    
    # Add convolutional layers with specified number of filters and activation function
    x = layers.Conv2D(filters[0], (3,3), activation="relu", strides=2, padding="same")(encoder_inputs)
    encoder_layers_dim.append(tuple(x.shape[1:]))  # Add conv layer dimensions to list
    
    # Add additional convolutional layers with specified number of filters and activation function
    mid_layers = [layers.Conv2D(f, 3, activation="relu", strides=2, padding="same") for f in filters[1:]]
    for mid_layer in mid_layers:
        x = mid_layer(x)
        encoder_layers_dim.append(tuple(x.shape[1:]))  # Add mid layer dimensions to list
    
    # Flatten convolutional output to prepare for dense layers
    x = layers.Flatten()(x)
    encoder_layers_dim.append(tuple(x.shape[1:]))  # Add flattened layer dimensions to list
    
    # Add dense layer with specified number of neurons and activation function
    x = layers.Dense(dense_layer_dim, activation='relu')(x)
    
    # Add output layers for latent space (mean and variance) and sample from this space
    z_mean = layers.Dense(latent_dim, name = "z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder_layers_dim.append(tuple(z.shape[1:]))  # Add output layer dimensions to list
    
    # Create encoder model
    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder'), encoder_layers_dim

# decoder model for grayscale images
def decoder_model_gs(encoder_layers_dim):
    # Extract necessary dimensions from encoder model output
    latent_dim = encoder_layers_dim[-1][0]
    dense_layer_dim = encoder_layers_dim[-2][0]
    first_conv_layer_dim = encoder_layers_dim[-3]
    output_layer = encoder_layers_dim[0]

    # Create input layer for latent space vector
    latent_inputs = keras.Input(shape=(latent_dim,))

    # Determine number of filters for each transpose convolutional layer
    filters = [f[-1] for f in encoder_layers_dim[1:-2]]

    # Feed latent vector through a dense layer with ReLU activation
    # Note that we apply the first filter in the form of dense and reshape it
    x = layers.Dense(dense_layer_dim, activation="relu")(latent_inputs)
    x = layers.Reshape(first_conv_layer_dim)(x)

    # Apply series of transpose convolutional layers with ReLU activation and same padding and Upsampling
    mid_layers = [layers.Conv2DTranspose(f, 3, activation="relu", strides=2, padding="same") for f in filters[::-1]]
    for mid_layer in mid_layers:
        x = mid_layer(x)

    # Apply final convolutional layer with sigmoid activation to output reconstructed image
    decoder_outputs = layers.Conv2DTranspose(output_layer[-1], 3, activation="sigmoid", padding="same")(x)

    # Create and return Keras model with latent vector as input and reconstructed image as output
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")


# VAE for GrayScale Images
class VAE_GrayScale(keras.Model):
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
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

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
        with tf.GradientTape() as tape:
            # Forward pass through encoder and decoder
            z_mean, z_log_var, z, reconstruction = self(data)
            
            # Compute reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )

            # Compute KL divergence loss
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

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


# callbacks
class VAECallback(tf.keras.callbacks.Callback):
    """
    randomly sample 10 images from test_data set and shows the reconsttruction after each epoch
    """
    def __init__(self, vae, test_dataset, log_dir, n=10):
        self.vae = vae
        self.test_dataset = test_dataset
        self.n = n
        self.log_dir = log_dir
    def on_epoch_end(self, epoch, logs=None):
        # Generate decoded images from the test input
        test_batch = next(iter(self.test_dataset))
        _, _, _, reconstructed_images = self.vae.predict(test_batch)

        # Rescale pixel values to [0, 1]
        # reconstructed_images = reconstructed_images
        reconstructed_images = np.clip(reconstructed_images, 0.0, 1.0)

        # Plot the original and reconstructed images side by side
        plt.figure(figsize=(10, 20))
        for i in range(self.n):
            plt.subplot(10, 2, 2*i+1)
            plt.imshow(test_batch[i], cmap='gray')
            plt.axis('off')
            plt.subplot(10, 2, 2*i+2)
            plt.imshow(reconstructed_images[i], cmap='gray')
            plt.axis('off')
        plt.savefig(self.log_dir + '/decoded_images_epoch_{:04d}.png'.format(epoch))
        # plt.show()
