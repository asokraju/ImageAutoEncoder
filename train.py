import os
import numpy as  np
import matplotlib.pyplot as plt
from PIL import  Image
import pandas
from matplotlib.pyplot import imshow
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
from tensorflow.keras.models import Model
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import layers

#
from Models import encoder_model_gs, decoder_model_gs, VAE_GrayScale, load_model_vae, VAECallback
from Data import get_image_data

#to reduce the tensorflow messages
# tf.get_logger().setLevel('WARNING')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')

# /home/kosaraju/anaconda3/envs/tf-gpu/bin/python $DIR/run.py --gamma=
#---------------------------------------------------------------------
if __name__ == '__main__':
    # The directory path where your image data is stored
    # all_dirs = ['/home/kkosara/AutoDRIVE-Nigel-Dataset/fishhook_30_hz', '/home/kkosara/AutoDRIVE-Nigel-Dataset/skidpad_30_hz', '/home/kkosara/AutoDRIVE-Nigel-Dataset/slalom_30_hz']
    DATA_DIR =  ['Data'] 
    OUTPUT_IMAGE_SHAPE = 56
    INPUT_SHAPE = (OUTPUT_IMAGE_SHAPE, OUTPUT_IMAGE_SHAPE, 1)
    FILTERS = [32, 64]
    NUM_CONV_LAYERS = len(FILTERS)
    DENSE_LAYER_DIM = 16
    LATENT_DIM = 6
    BETA = 1.0
    BATCH_SIZE = 128
    AUTOTUNE = tf.data.AUTOTUNE
    LEARNING_RATE = 1e-4
    PATIENCE = 10
    EPOCHS  = 2
    TRAIN_SPLIT = 0.8
    LOGDIR = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    GRAYSCALE = True
    
    # all_image_paths = get_image_data(DATA_DIR)
    # image_count = len(all_image_paths)

    # Create a dataset from the list of image file paths
    all_image_paths = get_image_data(DATA_DIR)
    image_count = len(all_image_paths)

    # Create a dataset from the list of image file paths
    dataset = tf.data.Dataset.from_tensor_slices(all_image_paths)

    # Shuffle the dataset with a buffer size of 1000 and seed of 42
    dataset = dataset.shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=False)

    # Load and preprocess the images
    # Function to load and preprocess an image
    def load_and_preprocess_image(path=all_image_paths, shape=OUTPUT_IMAGE_SHAPE, grayscale=GRAYSCALE):
        # Load the image file
        image = tf.io.read_file(path)
        # # Decode the image to a tensor
        image = tf.image.decode_jpeg(image, channels=3)
        # if grayscale:
            # Convert RGB to grayscale
        image = tf.image.rgb_to_grayscale(image)
        # # Resize the image to the desired size
        image = tf.image.resize(image, [shape, shape])
        # # Normalize the pixel values to be between 0 and 1
        image = tf.cast(image, tf.float32) / 255.0
        print(image.shape)
        return image
    dataset = dataset.map(load_and_preprocess_image)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    encoder, encoder_layers_dim = encoder_model_gs(input_shape = INPUT_SHAPE, filters=FILTERS, dense_layer_dim=DENSE_LAYER_DIM, latent_dim=LATENT_DIM)
    print(encoder.summary())
    print(encoder_layers_dim)
    decoder = decoder_model_gs(encoder_layers_dim)
    print(decoder.summary())
    vae = VAE_GrayScale(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam())
    
    
    #Callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGDIR)
    early_stopping_callback = EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    # Define callbacks to save encoder and decoder models
    model_checkpoint_callback = ModelCheckpoint(filepath=LOGDIR + "/vae-{epoch:02d}", save_best_only=True, save_format="tf")
    checkpoint_encoder = ModelCheckpoint(filepath=LOGDIR + "/encoder_weights-{epoch:02d}", save_weights_only=True, save_best_only=True)
    checkpoint_decoder = ModelCheckpoint(filepath=LOGDIR + "/decoder_weights-{epoch:02d}", save_weights_only=True, save_best_only=True)
    vae_callback = VAECallback(vae, dataset.take(10),log_dir=LOGDIR,)

    history = vae.fit(
        dataset,
        epochs=EPOCHS,
        callbacks=[vae_callback, tensorboard_callback, early_stopping_callback, model_checkpoint_callback, checkpoint_encoder, checkpoint_decoder],
    )
    

    encoder.save(LOGDIR +"/encoder", overwrite=True, save_format=None)
    decoder.save(LOGDIR +"/decoder", overwrite=True, save_format=None)
    vae.save(LOGDIR +"/vae", overwrite=True, save_format=None)


    ##
    # Split the dataset into training and validation sets
    # train_dataset = dataset.take(int(TRAIN_SPLIT * image_count))
    # train_dataset = train_dataset.shuffle(buffer_size=500, seed=42, reshuffle_each_iteration=False)
    # train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    # val_dataset = dataset.skip(int(TRAIN_SPLIT * image_count))
    # val_dataset = val_dataset.shuffle(buffer_size=500, seed=42, reshuffle_each_iteration=False)
    # val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    # Fit the VAE model with the DecodeCallback
    # # Split the dataset into training and validation sets
    # train_dataset = dataset.take(int(TRAIN_SPLIT * image_count))
    # train_dataset = train_dataset.shuffle(buffer_size=500, seed=42, reshuffle_each_iteration=False)
    # train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    # val_dataset = dataset.skip(int(TRAIN_SPLIT * image_count))
    # val_dataset = val_dataset.shuffle(buffer_size=500, seed=42, reshuffle_each_iteration=False)
    # val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    #Callbacks
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGDIR)
    # # early_stopping_callback = EarlyStopping(patience=10, restore_best_weights=True)
    # # early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # early_stopping_callback = EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    # # Define callbacks to save encoder and decoder models
    # # model_checkpoint_callback = ModelCheckpoint(filepath=LOGDIR+"/model_checkpoint.h5", save_best_only=True, save_weights_only=False, monitor="val_loss")
    # model_checkpoint_callback = ModelCheckpoint(filepath=LOGDIR + "/vae-{epoch:02d}", save_best_only=True, save_format="tf")
    # checkpoint_encoder = ModelCheckpoint(filepath=LOGDIR + "/encoder_weights-{epoch:02d}", save_weights_only=True, save_best_only=True)
    # checkpoint_decoder = ModelCheckpoint(filepath=LOGDIR + "/decoder_weights-{epoch:02d}", save_weights_only=True, save_best_only=True)


    # # Train the model with callbacks
    # # vae.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[checkpoint_callback, early_stop_callback])
    # history = vae.fit(
    #     train_dataset,
    #     epochs=EPOCHS,
    #     validation_data=val_dataset,
    #     callbacks=[tensorboard_callback, early_stopping_callback, model_checkpoint_callback, checkpoint_encoder, checkpoint_decoder],
    # )