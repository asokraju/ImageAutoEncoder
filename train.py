import os
import tensorflow as tf
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
from tensorflow.keras.models import Model
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

#
from Models import Encoder, Decoder, BetaVAE, load_model
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
    DATA_DIR = ['Data']
    OUTPUT_IMAGE_SHAPE = 200
    NUM_CONV_LAYERS = 3
    assert OUTPUT_IMAGE_SHAPE%(2**NUM_CONV_LAYERS)==0, "OUTPUT_IMAGE_SHAPE should be a multiple of 2^NUM_CONV_LAYERS, i.e. OUTPUT_IMAGE_SHAPE%(2**NUM_CONV_LAYERS)==0. This will ensure the Encoder and Decoder networks to have mirror image layers"
    LATENT_DIM = 8
    BETA = 4.0
    BATCH_SIZE = 100
    AUTOTUNE = tf.data.AUTOTUNE
    LEARNING_RATE = 1e-4
    PATIENCE = 10
    EPOCHS  = 1
    TRAIN_SPLIT = 0.8
    LOGDIR = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    NO_IMAGES_TO_PLOT = 2
    assert NO_IMAGES_TO_PLOT<BATCH_SIZE, "NO_IMAGES_TO_PLOT should be less than the batch size BATCH_SIZE"

    all_image_paths = get_image_data(DATA_DIR)
    image_count = len(all_image_paths)

    # Create a dataset from the list of image file paths
    dataset = tf.data.Dataset.from_tensor_slices(all_image_paths)

    # Shuffle the dataset with a buffer size of 1000 and seed of 42
    dataset = dataset.shuffle(buffer_size=500, seed=42, reshuffle_each_iteration=False)

    # Load and preprocess the images
    # Function to load and preprocess an image
    def load_and_preprocess_image(path=all_image_paths, shape=OUTPUT_IMAGE_SHAPE):
        # Load the image file
        image = tf.io.read_file(path)
        # # Decode the image to a tensor
        image = tf.image.decode_jpeg(image, channels=3)
        # # Resize the image to the desired size
        image = tf.image.resize(image, [shape, shape])
        # # Normalize the pixel values to be between 0 and 1
        image = tf.cast(image, tf.float32) / 255.0
        return image
    dataset = dataset.map(load_and_preprocess_image)

    # Split the dataset into training and validation sets
    train_dataset = dataset.take(int(TRAIN_SPLIT * image_count))
    train_dataset = train_dataset.shuffle(buffer_size=500, seed=42, reshuffle_each_iteration=False)
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    val_dataset = dataset.skip(int(TRAIN_SPLIT * image_count))
    val_dataset = val_dataset.shuffle(buffer_size=500, seed=42, reshuffle_each_iteration=False)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)


    #Model
    encoder = Encoder(num_conv_layers=NUM_CONV_LAYERS, latent_dim=LATENT_DIM)
    decoder = Decoder(output_size=(OUTPUT_IMAGE_SHAPE, OUTPUT_IMAGE_SHAPE, 3), num_conv_layers=NUM_CONV_LAYERS, latent_dim=LATENT_DIM)
    vae = BetaVAE(encoder=encoder, decoder=decoder, beta=BETA)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    #Callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGDIR)
    # early_stopping_callback = EarlyStopping(patience=10, restore_best_weights=True)
    # early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    early_stopping_callback = EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    # Define callbacks to save encoder and decoder models
    # model_checkpoint_callback = ModelCheckpoint(filepath=LOGDIR+"/model_checkpoint.h5", save_best_only=True, save_weights_only=False, monitor="val_loss")
    model_checkpoint_callback = ModelCheckpoint(filepath=LOGDIR + "/vae-{epoch:02d}", save_best_only=True, save_format="tf")
    checkpoint_encoder = ModelCheckpoint(filepath=LOGDIR + "/encoder_weights-{epoch:02d}", save_weights_only=True, save_best_only=True)
    checkpoint_decoder = ModelCheckpoint(filepath=LOGDIR + "/decoder_weights-{epoch:02d}", save_weights_only=True, save_best_only=True)


    # Train the model with callbacks
    # vae.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[checkpoint_callback, early_stop_callback])
    vae.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[tensorboard_callback, early_stopping_callback, model_checkpoint_callback, checkpoint_encoder, checkpoint_decoder],
    )

    encoder.save_weights(LOGDIR + "/encoder_weights.h5")
    decoder.save_weights(LOGDIR + "/decoder_weights.h5")

    assert NO_IMAGES_TO_PLOT<image_count, "NO_IMAGES_TO_PLOT should be less than the avaialble images image_count"
    loaded_encoder, loaded_decoder, loaded_vae = load_model(train_dataset, LOGDIR, NUM_CONV_LAYERS, LATENT_DIM, OUTPUT_IMAGE_SHAPE, BETA, LEARNING_RATE, n=NO_IMAGES_TO_PLOT, plot= True)