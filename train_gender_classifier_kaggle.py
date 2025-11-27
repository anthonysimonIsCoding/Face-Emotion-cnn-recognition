"""
File: train_gender_classifier.py
Author: Adapted from Octavio Arriaga's original code
Description: Train gender classification model using TensorFlow 2.x / Keras for Python 3.10+
Updated to use tf.keras.utils.image_dataset_from_directory for loading from folder structure
instead of DataManager/ImageGenerator/split_imdb_data, and modern Keras APIs (tf.data for efficiency,
no fit_generator). Assumes folder structure: datasets/gender/<class_folders>/ (e.g., Female Faces/, Male Faces/).
Uses validation_split=0.2 for validation data. Advanced augmentation mimics original ImageGenerator
(saturation/brightness/contrast/lighting noise/horizontal flip, no vertical flip, optional random crop)
using tf.keras.layers.Random*. Preprocessing normalizes to [-1, 1] range. Grayscale=True.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, SeparableConv2D, 
    MaxPooling2D, GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Parameters
batch_size = 32
num_epochs = 1000
validation_split = 0.2
do_random_crop = False  # Set to True to enable random crop augmentation
patience = 100
num_classes = 2  # Gender: 0=Female, 1=Male (auto-detected from folders)
input_shape = (64, 64, 1)
grayscale = True  # Enforced for input_shape
base_path = '../trained_models/gender_models/'
data_dir = 'datasets/gender'  # Root folder with subfolders: Female Faces/, Male Faces/

def preprocess_input(x):
    """Custom preprocessing: normalize to [-1, 1] range."""
    x = tf.cast(x, tf.float32) / 255.0
    x = x - 0.5
    x = x * 2.0
    return x

def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # Base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(num_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)

    model = Model(img_input, output)
    return model

# Augmentation layers mimicking ImageGenerator (saturation via hue, brightness/contrast, lighting noise, hflip, optional crop)
aug_layers = tf.keras.Sequential([
    layers.RandomRotation(factor=0.0175, input_shape=input_shape),  # ~10 degrees rotation
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),  # Shift
    layers.RandomZoom(height_factor=0.1, width_factor=0.1),  # Zoom range ~0.1
    layers.RandomFlip("horizontal"),  # Horizontal flip (probability ~0.5)
    # Color jitter (original: saturation/brightness/contrast vars=0.5)
    layers.RandomBrightness(factor=0.2),  # Brightness var approx
    layers.RandomContrast(factor=0.2),  # Contrast var approx
    layers.Lambda(lambda x: tf.image.random_hue(x, max_delta=0.1)),  # Saturation approx via hue
    # Lighting noise (simple Gaussian approx original PCA, std~0.5 scaled)
    layers.GaussianNoise(stddev=25.5),  # ~0.1 * 255
    # Optional random crop (if do_random_crop=True)
])
if do_random_crop:
    aug_layers.add(layers.RandomCrop(height=56, width=56))  # Crop to 56x56
    aug_layers.add(layers.Resizing(64, 64))  # Resize back

# Load datasets using tf.keras.utils.image_dataset_from_directory (modern, efficient replacement for ImageDataGenerator)
def load_dataset(data_dir, batch_size, is_training=True, validation_split=None, subset=None):
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=input_shape[:2],
        batch_size=batch_size,
        label_mode='categorical',
        color_mode='grayscale' if grayscale else 'rgb',
        validation_split=validation_split,
        subset=subset,
        seed=123,  # For reproducibility
        shuffle=True if is_training else False
    )
    # Apply preprocess
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    # Apply aug only for training
    if is_training:
        ds = ds.map(lambda x, y: (aug_layers(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.cache().prefetch(tf.data.AUTOTUNE)

# Load train/val datasets (split 80/20)
train_ds = load_dataset(data_dir, batch_size, is_training=True, validation_split=validation_split, subset='training')
val_ds = load_dataset(data_dir, batch_size, is_training=False, validation_split=validation_split, subset='validation')

print('Number of training samples:', len(train_ds) * batch_size)  # Approx
print('Number of validation samples:', len(val_ds) * batch_size)  # Approx

# Model creation and compilation
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks (updated metrics/paths)
log_file_path = os.path.join(base_path, 'gender_training.log')
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/2), verbose=1)
trained_models_path = os.path.join(base_path, 'gender_mini_XCEPTION')
model_names = f'{trained_models_path}.{{epoch:02d}}-{{val_accuracy:.2f}}.h5'  # Updated: val_accuracy, .h5
model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

print('Training dataset: gender (from directory)')

# Training (modern fit with tf.data)
model.fit(
    train_ds,
    epochs=num_epochs,
    verbose=1,
    callbacks=callbacks,
    validation_data=val_ds
)

print('Training completed.')