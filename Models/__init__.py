
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
    

# Define encoder model
def encoder_model(input_shape, filters, dense_layer_dim, latent_dim):
    # Create input layer
    encoder_inputs = keras.Input(shape=input_shape)
    
    # Add convolutional layers with specified number of filters and activation function
    x = layers.Conv2D(filters[0], (3,3), activation="relu", strides=2, padding="same")(encoder_inputs)
    x = MaxPooling2D((2, 2), padding="same")(x)
    # Add additional convolutional layers with specified number of filters and activation function
    mid_layers = [layers.Conv2D(f, 3, activation="relu", strides=2, padding="same") for f in filters[1:]]
    for mid_layer in mid_layers:
        x = mid_layer(x)
        x = MaxPooling2D((2, 2), padding="same")(x)
    
    # Flatten convolutional output to prepare for dense layers
    x = layers.Flatten()(x)
    
    # Add dense layer with specified number of neurons and activation function
    x = layers.Dense(dense_layer_dim, activation='relu')(x)
    
    # Add output layers for latent space (mean and variance) and sample from this space
    z_mean = layers.Dense(latent_dim, name = "z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    
    # Create encoder model
    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

# decoder
def decoder_model(output_shape, filters, dense_layer_dim, latent_dim):
    # define input layer for the latent vector
    latent_inputs = keras.Input(shape=(latent_dim,))
    
    x = layers.Dense(dense_layer_dim, activation='relu')(latent_inputs)

    # filters are applied in reverse order compared to the encoder
    filters_reverse = filters[::-1]
    # calculate dimensions of first convolutional layer based on output shape and number of filters
    num_conv_layers = len(filters)
    first_layer_dim = (output_shape[0]//(2**(2*num_conv_layers)), output_shape[1]//(2**(2*num_conv_layers)), filters_reverse[0])

    # feed latent vector through a dense layer with ReLU activation
    # note that we apply the first filter in the form of dense and reshape it
    x = layers.Dense(first_layer_dim[0] * first_layer_dim[1] * first_layer_dim[2], activation="relu")(x)
    # reshape output from dense layer to match dimensions of first convolutional layer
    x = layers.Reshape(first_layer_dim)(x)
    x = UpSampling2D((2, 2))(x)
    # apply series of transpose convolutional layers with ReLU activation and same padding and Upsampling
    mid_layers = [layers.Conv2DTranspose(f, 3, activation="relu", strides=2, padding="same") for f in filters_reverse[1:]]
    for mid_layer in mid_layers:
        x = mid_layer(x)
        x = UpSampling2D((2, 2))(x)
    
    # apply final convolutional layer with sigmoid activation to output reconstructed image
    decoder_outputs = layers.Conv2DTranspose(output_shape[2], 3,strides=2, activation="sigmoid", padding="same")(x)
    
    # create and return Keras model with latent vector as input and reconstructed image as output
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")



class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, z, reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # z_mean, z_log_var, z = self.encoder(data)
            # reconstruction = self.decoder(z)
            z_mean, z_log_var, z, reconstruction = self(data)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def plot_latent_space(vae, digit_size=200, scale=1, n=3, figsize=30):
    # display a n*n 2D manifold of digits
    # digit_size = 28
    # scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


class Encoder(Model):
    def __init__(self, num_conv_layers=2, latent_dim=8):
        super(Encoder, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.conv_layers = []
        for i in range(num_conv_layers):
            conv_layer = Conv2D(32 * (2**i), (3, 3), activation='relu', strides=2, padding='same')
            self.conv_layers.append(conv_layer)
        self.flatten = Flatten()
        self.dense1 = Dense(latent_dim)
        
    def call(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return x

class Decoder(Model):
    def __init__(self, output_size=(400, 400, 3), num_conv_layers=2, latent_dim=8):
        super(Decoder, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.dense1 = Dense((output_size[0]//(2**self.num_conv_layers)) * (output_size[1]//(2**self.num_conv_layers)) * 32 * (2**(num_conv_layers-1)), activation='relu')
        self.reshape1 = Reshape((output_size[0]//(2**self.num_conv_layers), output_size[1]//(2**self.num_conv_layers), 32 * (2**(num_conv_layers-1))))
        self.conv_layers = []
        for i in range(num_conv_layers-1):
            conv_layer = Conv2DTranspose(32 * (2**(num_conv_layers-2-i)), (3, 3), activation='relu', strides=2, padding='same')
            self.conv_layers.append(conv_layer)
        self.conv3 = Conv2DTranspose(output_size[2], (3, 3), activation='sigmoid', strides=2, padding='same')
        
    def call(self, x):
        x = self.dense1(x)
        x = self.reshape1(x)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)        
        x = self.conv3(x)
        return x

class BetaVAE(Model):
    def __init__(self, encoder, decoder, beta=4.0, mse_recon_ratio=0.5):
        super(BetaVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.mse_recon_ratio = mse_recon_ratio
    def call(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, z
    
    # def train_step(self, data):
    #     with tf.GradientTape() as tape:
    #         recon_x, z = self(data) # self.call(data)
    #         reconstruction_loss = tf.reduce_mean(tf.square(data - recon_x))
    #         kl_loss = -0.5 * self.beta * tf.reduce_mean(1 + tf.math.log(tf.square(z)) - tf.square(z))
    #         total_loss = reconstruction_loss + kl_loss
    #     grads = tape.gradient(total_loss, self.trainable_weights)
    #     self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    #     return {'loss': total_loss}
    def train_step(self, data):
        with tf.GradientTape() as tape:
            recon_x, z = self(data) # self.call(data)
            mse_reconstruction_loss = tf.reduce_mean(tf.square(data - recon_x))
            ce_reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(data, recon_x), axis=[1, 2])
            reconstruction_loss = self.mse_recon_ratio * mse_reconstruction_loss + (1 - self.mse_reconstruction_loss) * ce_reconstruction_loss
            kl_loss = -0.5 * self.beta * tf.reduce_mean(1 + tf.math.log(tf.square(z)) - tf.square(z))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': total_loss}




def load_model(dataset, LOGDIR, NUM_CONV_LAYERS, LATENT_DIM, OUTPUT_IMAGE_SHAPE, BETA, LEARNING_RATE, n=5, plot= True):
    #Model
    encoder = Encoder(num_conv_layers=NUM_CONV_LAYERS, latent_dim=LATENT_DIM)
    decoder = Decoder(output_size=(OUTPUT_IMAGE_SHAPE, OUTPUT_IMAGE_SHAPE, 3), num_conv_layers=NUM_CONV_LAYERS, latent_dim=LATENT_DIM)
    vae = BetaVAE(encoder=encoder, decoder=decoder, beta=BETA)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))



    # load the model
    encoder_checkpoint_path = LOGDIR + "\encoder_weights-01.index"
    encoder.load_weights(encoder_checkpoint_path)

    decoder_checkpoint_path = LOGDIR + "\decoder_weights-01.index"
    decoder.load_weights(decoder_checkpoint_path)
    if plot:
        plt.figure(figsize=(10, 4))
        images = list(dataset.take(n))[0]
        encoded_imgs = encoder.predict(images)
        decoded_imgs = decoder.predict(encoded_imgs)
        # print(images.shape, images[0].shape)
        # dataset_batch = next(iter(dataset.batch(n)))
        # print(dataset_batch.shape)
        # encoded_imgs = encoder.predict(dataset_batch)
        # decoded_imgs = decoder.predict(encoded_imgs)
        # print(decoded_imgs.shape)
        print(images[0])
        print(decoded_imgs[0])
        for i in range(n):
            # Display original images
            ax = plt.subplot(2, n, i + 1)
            # print(list(dataset.take(1))[0])
            plt.imshow(images[i])
            plt.axis('off')

            # Display decoder-generated images
            ax = plt.subplot(2, n, i + n + 1)
            plt.imshow(decoded_imgs[i])
            plt.axis('off')
        plt.savefig(LOGDIR+'/validation.jpg')
        plt.show()


    return encoder, decoder, vae

def load_model_vae(dataset, LOGDIR, encoder, decoder, vae, n=5, plot= True):
    
    
    # #Model
    # encoder = tf.keras.saving.load_model(LOGDIR +"/encoder") #encoder_model(INPUT_SHAPE, FILTERS, DENSE_LAYER_DIM, LATENT_DIM)
    # decoder = tf.keras.saving.load_model(LOGDIR +"/decoder") #decoder_model(INPUT_SHAPE, FILTERS, LATENT_DIM)
    # vae = tf.keras.saving.load_model(LOGDIR +"/vae") #VAE(encoder=encoder, decoder=decoder)
    # vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))



    # # load the model
    # encoder_checkpoint_path = LOGDIR + "\encoder_weights-01.index"
    # encoder.load_weights(encoder_checkpoint_path)

    # decoder_checkpoint_path = LOGDIR + "\decoder_weights-01.index"
    # decoder.load_weights(decoder_checkpoint_path)
    if plot:
        plt.figure(figsize=(10, 4))
        images = list(dataset.take(n))[0]
        # images = np.expand_dims(images, axis=0)  
        print("images shape:", images.shape)
        encoded_imgs = encoder.predict(np.expand_dims(images, axis=0) )
        print(encoded_imgs, encoded_imgs[0].shape)
        decoded_imgs = [decoder.predict(encoded_img) for encoded_img in encoded_imgs ]
        print("no of decoded_imgs", len(decoded_imgs), decoded_imgs[0].shape)
        # print(images.shape, images[0].shape)
        # dataset_batch = next(iter(dataset.batch(n)))
        # print(dataset_batch.shape)
        # encoded_imgs = encoder.predict(dataset_batch)
        # decoded_imgs = decoder.predict(encoded_imgs)
        # print(decoded_imgs.shape)
        print(decoded_imgs[0])
        for i in range(n):
            plt.imshow(images[i])
            # plt.imshow(np.squeeze(decoded_imgs[i]))
            # # Display original images
            # ax = plt.subplot(2, n, i + 1)
            # # print(list(dataset.take(1))[0])
            # plt.imshow(images[i])
            # plt.axis('off')

            # # Display decoder-generated images
            # ax = plt.subplot(2, n, i + n + 1)
            # plt.imshow(np.squeeze(decoded_imgs[i]))
            # plt.axis('off')
        plt.savefig(LOGDIR+'/validation.jpg')
        plt.show()


    return encoder, decoder, vae


# # Define a custom callback for image visualization
# class ImageVisualizationCallback(tf.keras.callbacks.Callback):
#     def __init__(self, val_data, log_dir):
#         super().__init__()
#         self.val_data = val_data
#         self.log_dir = log_dir

#     def on_epoch_end(self, epoch, logs=None):
#         # Select 5 random images from the validation dataset
#         val_images = self.val_data.take(5)
        
#         # Generate reconstructed images using the trained model
#         reconstructed_images = self.model.predict(val_images)

#         # Save the reconstructed images to TensorBoard
#         file_writer = tf.summary.create_file_writer(self.log_dir)
#         with file_writer.as_default():
#             tf.summary.image("Reconstructed Images - Epoch {}".format(epoch), reconstructed_images, max_outputs=5, step=0)
    





# OUTPUT_IMAGE_SHAPE = 64

# class Encoder(tf.keras.Model):
#     def __init__(self, latent_dim=8, num_layers=2, layer_param=32):
#         super(Encoder, self).__init__()
#         self.num_layers = num_layers
#         self.layer_param = layer_param
#         self.conv_layers = [Conv2D(layer_param, (3, 3), activation='relu', strides=2, padding='same') for _ in range(num_layers)]
#         self.flatten = Flatten()
#         self.dense1 = Dense(latent_dim)
        
#     def call(self, x):
#         for i in range(self.num_layers):
#             x = self.conv_layers[i](x)
#         x = self.flatten(x)
#         x = self.dense1(x)
#         return x

# class Decoder(tf.keras.Model):
#     def __init__(self, output_size=(OUTPUT_IMAGE_SHAPE, OUTPUT_IMAGE_SHAPE, 3), latent_dim=8, num_layers=2, layer_param=32):
#         super(Decoder, self).__init__()
#         self.num_layers = num_layers
#         self.layer_param = layer_param
#         self.dense1 = Dense(output_size[0]//4 * output_size[1]//4 * layer_param, activation='relu')
#         self.reshape1 = Reshape((output_size[0]//4, output_size[1]//4, layer_param))
#         self.conv_layers = [Conv2DTranspose(layer_param, (3, 3), activation='relu', strides=2, padding='same') for _ in range(num_layers-1)]
#         self.conv_layers.append(Conv2DTranspose(output_size[2], (3, 3), activation='sigmoid', padding='same'))
        
#     def call(self, x):
#         x = self.dense1(x)
#         x = self.reshape1(x)
#         for i in range(self.num_layers):
#             x = self.conv_layers[i](x)
#         return x

# class Autoencoder(tf.keras.Model):
#     def __init__(self, latent_dim=8, num_layers=2, layer_param=32):
#         super(Autoencoder, self).__init__()
#         self.encoder = Encoder(latent_dim, num_layers, layer_param)
#         self.decoder = Decoder(latent_dim=latent_dim, num_layers=num_layers, layer_param=layer_param)
        
#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded
