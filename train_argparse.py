import os
import numpy as  np
import matplotlib.pyplot as plt
from PIL import  Image
import pandas
from matplotlib.pyplot import imshow
from datetime import datetime
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
from tensorflow.keras.models import Model
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
from Models import encoder_model_gs, decoder_model_gs, VAE_GrayScale, load_model_vae, VAECallback
from Data import get_image_data


#to reduce the tensorflow messages
# tf.get_logger().setLevel('WARNING')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')

# /home/kosaraju/anaconda3/envs/tf-gpu/bin/python $DIR/run.py --gamma=
#---------------------------------------------------------------------

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
if __name__ == '__main__':
    # The directory path where your image data is stored
    # all_dirs = ['/home/kkosara/AutoDRIVE-Nigel-Dataset/fishhook_30_hz', '/home/kkosara/AutoDRIVE-Nigel-Dataset/skidpad_30_hz', '/home/kkosara/AutoDRIVE-Nigel-Dataset/slalom_30_hz']
    DATA_DIR =  [r'C:\\Users\\kkosara\\eight_30_hz'] # C:\Users\kkosara\eight_30_hz
    
    args = parse_arguments()
    
    OUTPUT_IMAGE_SHAPE = args.output_image_shape
    # INPUT_SHAPE = (OUTPUT_IMAGE_SHAPE, OUTPUT_IMAGE_SHAPE, 1)
    FILTERS = args.filters
    NUM_CONV_LAYERS = len(FILTERS)
    DENSE_LAYER_DIM = args.dense_layer_dim
    LATENT_DIM = args.latent_dim
    BETA = args.beta
    BATCH_SIZE = args.batch_size
    AUTOTUNE = tf.data.AUTOTUNE
    LEARNING_RATE = args.learning_rate
    PATIENCE = args.patience
    EPOCHS = args.epochs
    TRAIN_SPLIT = args.train_split
    VAL_SPLIT = 1- TRAIN_SPLIT
    INPUT_SHAPE = (OUTPUT_IMAGE_SHAPE, OUTPUT_IMAGE_SHAPE, 1)
    
    LOGDIR = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    GRAYSCALE = True
    
    train_datagen_args = dict(
        rescale = 1/255.,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

    train_datagen = ImageDataGenerator(**train_datagen_args)
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR[0],
        batch_size = BATCH_SIZE,
        shuffle = True,
        target_size = (OUTPUT_IMAGE_SHAPE, OUTPUT_IMAGE_SHAPE),
        class_mode = 'binary')
    # # all_image_paths = get_image_data(DATA_DIR)
    # # image_count = len(all_image_paths)

    # # Create a dataset from the list of image file paths
    # all_image_paths = get_image_data(DATA_DIR)
    # image_count = len(all_image_paths)

    # # Create a dataset from the list of image file paths
    # dataset = tf.data.Dataset.from_tensor_slices(all_image_paths)

    # # Shuffle the dataset with a buffer size of 1000 and seed of 42
    # dataset = dataset.shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=False)

    # # Load and preprocess the images
    # # Function to load and preprocess an image
    # def load_and_preprocess_image(path=all_image_paths, shape=OUTPUT_IMAGE_SHAPE, grayscale=GRAYSCALE):
    #     # Load the image file
    #     image = tf.io.read_file(path)
    #     # # Decode the image to a tensor
    #     image = tf.image.decode_jpeg(image, channels=3)
    #     # if grayscale:
    #         # Convert RGB to grayscale
    #     image = tf.image.rgb_to_grayscale(image)
    #     # # Resize the image to the desired size
    #     image = tf.image.resize(image, [shape, shape])
    #     # # Normalize the pixel values to be between 0 and 1
    #     image = tf.cast(image, tf.float32) / 255.0
    #     print(image.shape)
    #     return image
    
    # dataset = dataset.map(load_and_preprocess_image)
    # # Split the dataset into training and validation sets
    # train_dataset = dataset.take(int(TRAIN_SPLIT * image_count))
    # val_dataset = dataset.take(int((1-TRAIN_SPLIT) * image_count))
    # # train_dataset = train_dataset.shuffle(buffer_size=500, seed=42)
    # train_dataset = train_dataset.shuffle(buffer_size=500, seed=42)
    # train_dataset = train_dataset.batch(BATCH_SIZE)
    # train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    # val_dataset = val_dataset.shuffle(buffer_size=500, seed=42)
    # val_dataset = val_dataset.batch(BATCH_SIZE)
    # val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

    # encoder, encoder_layers_dim = encoder_model_gs(input_shape = INPUT_SHAPE, filters=FILTERS, dense_layer_dim=DENSE_LAYER_DIM, latent_dim=LATENT_DIM)
    # print(encoder.summary())
    # print(encoder_layers_dim)
    # decoder = decoder_model_gs(encoder_layers_dim)
    # print(decoder.summary())
    # vae = VAE_GrayScale(encoder, decoder)
    # vae.compile(optimizer=tf.keras.optimizers.Adam())
    
    
    # #Callbacks
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGDIR)
    # early_stopping_callback = EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    # # Define callbacks to save encoder and decoder models
    # model_checkpoint_callback = ModelCheckpoint(filepath=LOGDIR + "/vae-{epoch:02d}", save_best_only=True, save_format="tf")
    # checkpoint_encoder = ModelCheckpoint(filepath=LOGDIR + "/encoder_weights-{epoch:02d}", save_weights_only=True, save_best_only=True)
    # checkpoint_decoder = ModelCheckpoint(filepath=LOGDIR + "/decoder_weights-{epoch:02d}", save_weights_only=True, save_best_only=True)
    # vae_callback = VAECallback(vae, val_dataset,log_dir=LOGDIR,)

    # history = vae.fit(
    #     train_dataset,
    #     epochs=EPOCHS,
    #     callbacks=[vae_callback, tensorboard_callback, early_stopping_callback, model_checkpoint_callback, checkpoint_encoder, checkpoint_decoder],
    # )
    

    # encoder.save(LOGDIR +"/encoder", overwrite=True, save_format=None)
    # decoder.save(LOGDIR +"/decoder", overwrite=True, save_format=None)
    # vae.save(LOGDIR +"/vae", overwrite=True, save_format=None)
