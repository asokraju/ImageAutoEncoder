
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as  np

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

def load_model2(dataset, LOGDIR, NUM_CONV_LAYERS, LATENT_DIM, OUTPUT_IMAGE_SHAPE, BETA, LEARNING_RATE, n=5, plot= True):
    #Model
    encoder = Encoder(num_conv_layers=NUM_CONV_LAYERS, latent_dim=LATENT_DIM)
    decoder = Decoder(output_size=(OUTPUT_IMAGE_SHAPE, OUTPUT_IMAGE_SHAPE, 3), num_conv_layers=NUM_CONV_LAYERS, latent_dim=LATENT_DIM)
    vae = BetaVAE(encoder=encoder, decoder=decoder, beta=BETA)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    # load the model
    encoder_checkpoint_path = LOGDIR + "\encoder_weights"
    encoder.load_weights(encoder_checkpoint_path)

    decoder_checkpoint_path = LOGDIR + "\decoder_weights"
    decoder.load_weights(decoder_checkpoint_path)

    if plot:
        plt.figure(figsize=(10, 4))
        dataset_batch = next(iter(dataset.batch(n)))
        encoded_imgs = encoder.predict(dataset_batch)
        decoded_imgs = decoder.predict(encoded_imgs)
        print(decoded_imgs.shape)
        for i in range(n):
            # Display original images
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(dataset_batch[i])
            plt.axis('off')

            # Display decoder-generated images
            ax = plt.subplot(2, n, i + n + 1)
            plt.imshow(decoded_imgs[i])
            plt.axis('off')
        plt.show()
        plt.savefig(LOGDIR+'/validation.png')

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
