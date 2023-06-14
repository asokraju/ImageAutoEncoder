import os
import pandas as pd
import argparse
from datetime import datetime

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.backend import random_normal
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
# from tensorflow.keras import saving
from utils import encoder_model, decoder_model, VAE, get_image_data, VAECallback, TotalLoss

# ----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, help='Path to the image data', default= r'Data')
    parser.add_argument('--logs-dir', type=str, help='Path to store logs', default=r"logs")
    parser.add_argument('--output-image-shape', type=int, default=56)
    parser.add_argument('--filters', type=int, nargs='+', default=[32, 64])
    parser.add_argument('--dense-layer-dim', type=int, default=16)
    parser.add_argument('--latent-dim', type=int, default=6)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--train-split', type=float, default=0.8)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = parse_arguments()
    IMAGE_DIR = args.image_dir
    LOGS_DIR = args.logs_dir
    all_image_paths = get_image_data(IMAGE_DIR)
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

    LOGDIR = os.path.join(LOGS_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(LOGDIR)
    os.mkdir(LOGDIR)

    df_train = pd.DataFrame({'image_paths': all_image_paths[:int(image_count*TRAIN_SPLIT)]})
    df_test = pd.DataFrame({'image_paths': all_image_paths[int(image_count*TRAIN_SPLIT):]})

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
    vae.compile(optimizer=Adam(learning_rate=LEARNING_RATE), metrics=[TotalLoss()])
    vae_callback = VAECallback(vae, test_data_generator, LOGDIR)
    tensorboard_cb = TensorBoard(log_dir=LOGDIR, histogram_freq=1)
    vae_path = os.path.join(LOGDIR, "vae")
    os.mkdir(vae_path)
    # encoder_path = os.path.join(LOGDIR, "encoder")
    # decoder_path = os.path.join(LOGDIR, "decoder")
    checkpoint_cb = ModelCheckpoint(filepath=vae_path, save_weights_only=True, verbose=1)

    earlystopping_cb = EarlyStopping(
        monitor="total_loss",
        min_delta=1e-2,
        patience=5,
        verbose=1,
    )

    history = vae.fit(
        train_data_generator,
        epochs=EPOCHS,
        validation_data=test_data_generator,
        callbacks=[tensorboard_cb, vae_callback, checkpoint_cb]
    )