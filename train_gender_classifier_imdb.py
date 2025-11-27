# train_gender_classifier_imdb_singlefile.py
"""
Single-file trainer for IMDB_crop gender dataset (Python 3.10 + TF2).
- No external utils imports.
- Reads imdb .mat, filters samples (face_score > 3, single face, gender known).
- Full augmentation preserved (saturation/brightness/contrast/lighting noise/flip/rotation/random crop).
- Model: mini_XCEPTION (tf.keras).
- Uses generator -> model.fit (no fit_generator).
Requirements:
    pip install tensorflow opencv-python scipy numpy
Run:
    python train_gender_classifier_imdb_singlefile.py
"""

import os
import time
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import cv2
from scipy.io import loadmat
import scipy.ndimage as ndi

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, SeparableConv2D,
    MaxPooling2D, GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# -----------------------
# CONFIG (tùy chỉnh được)
# -----------------------
batch_size = 32
num_epochs = 1000
validation_split = 0.2
do_random_crop = False        # can set True for random crop augmentation
patience = 100
num_classes = 2               # 0=female,1=male
input_shape = (64, 64, 1)     # (h, w, c)
grayscale = True
images_path = 'datasets/imdb_crop/'  # where images live (prefix to paths in .mat)
mat_path = 'datasets/imdb_crop/imdb.mat'       # path to imdb .mat file
base_path = 'models/gender_model/'
os.makedirs(base_path, exist_ok=True)

# -----------------------
# Preprocessing utilities
# -----------------------
def preprocess_input(x: np.ndarray) -> np.ndarray:
    """Normalize image array in [0..255] to [-1,1] float32"""
    x = x.astype(np.float32) / 255.0
    x = (x - 0.5) * 2.0
    return x

def ensure_gray(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale if needed; return HxW"""
    if img is None:
        return None
    if img.ndim == 3:
        # OpenCV loads BGR; convert to RGB then to gray by cv2
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# -----------------------
# Load imdb .mat -> mapping image_path -> gender
# -----------------------
def load_imdb_mat(mat_path: str, images_path_prefix: str = '') -> Dict[str, int]:
    """
    Load imdb .mat and return dict mapping image_file_path -> gender (0=Female,1=Male).
    Applies masks:
      - face_score > 3
      - second_face_score is NaN (only one face)
      - gender is not NaN
    Attempts to resolve image path using images_path_prefix if needed.
    """
    print(f'Loading .mat from: {mat_path}')
    dataset = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    if 'imdb' not in dataset:
        # sometimes loadmat returns a dict with top-level keys, or nested differently
        # attempt alternative access:
        raise ValueError('No "imdb" key in .mat file. Check file structure.')
    imdb = dataset['imdb']

    # Possible shapes: imdb.full_path may be an array of strings or nested arrays
    try:
        image_paths_raw = imdb['full_path']
        genders_raw = imdb['gender']
        face_scores = np.array(imdb['face_score'], dtype=np.float32)
        second_face_scores = np.array(imdb['second_face_score'], dtype=np.float32)
    except Exception:
        # try attribute-style
        image_paths_raw = imdb.full_path
        genders_raw = imdb.gender
        face_scores = np.array(imdb.face_score, dtype=np.float32)
        second_face_scores = np.array(imdb.second_face_score, dtype=np.float32)

    # Normalize image_paths_raw to flat numpy array of strings
    def normalize_paths(arr):
        out = []
        if isinstance(arr, np.ndarray) or isinstance(arr, list):
            for el in arr:
                if isinstance(el, (np.ndarray, list)) and len(el) > 0:
                    # sometimes elements are array containing single string
                    val = el[0] if len(el) > 0 else ''
                else:
                    val = el
                out.append(str(val))
        else:
            out.append(str(arr))
        return np.array(out, dtype=object)

    image_paths = normalize_paths(image_paths_raw)
    genders = np.array(genders_raw, dtype=np.float32)

    # Masks
    face_mask = face_scores > 3.0
    second_face_mask = np.isnan(second_face_scores)
    gender_known_mask = ~np.isnan(genders)
    mask = np.logical_and(face_mask, second_face_mask)
    mask = np.logical_and(mask, gender_known_mask)

    image_paths = image_paths[mask]
    genders = genders[mask].astype(int).tolist()

    mapping = {}
    for i, p in enumerate(image_paths):
        p = str(p)
        # candidate paths
        candidates = [
            p,
            os.path.join(images_path_prefix, p),
            os.path.join(images_path_prefix, os.path.basename(p)),
        ]
        final = None
        for c in candidates:
            if os.path.isfile(c):
                final = c
                break
        # if none exists, still store the candidate with prefix (user might fix paths)
        if final is None:
            continue
        mapping[final] = genders[i]

    return mapping

# -----------------------
# Split
# -----------------------
def split_imdb_data(ground_truth_data: Dict[str,int], validation_split: float = 0.2, seed: int = 123):
    keys = sorted(list(ground_truth_data.keys()))
    random.Random(seed).shuffle(keys)
    n_val = int(len(keys) * validation_split)
    val_keys = keys[:n_val]
    train_keys = keys[n_val:]
    return train_keys, val_keys

# -----------------------
# Image augmentation utilities (FULL)
# -----------------------
def color_jitter(image_array: np.ndarray, saturation_var=0.5, brightness_var=0.5, contrast_var=0.5):
    """Apply saturation, brightness, contrast jitter on RGB image_array (H,W,3), values [0..255]."""
    img = image_array.astype(np.float32)
    # saturation
    if saturation_var:
        gray = img.dot([0.299, 0.587, 0.114])
        alpha = 2.0 * np.random.random() * saturation_var
        alpha = alpha + 1 - saturation_var
        img = (alpha * img + (1 - alpha) * gray[:, :, None])
    # brightness
    if brightness_var:
        alpha = 2.0 * np.random.random() * brightness_var
        alpha = alpha + 1 - brightness_var
        img = img * alpha
    # contrast
    if contrast_var:
        gs = (img.dot([0.299, 0.587, 0.114]).mean()) * np.ones_like(img)
        alpha = 2.0 * np.random.random() * contrast_var
        alpha = alpha + 1 - contrast_var
        img = img * alpha + (1 - alpha) * gs
    return np.clip(img, 0, 255).astype(np.uint8)

def add_lighting_noise(image_array: np.ndarray, lighting_std=0.5):
    """
    PCA-based lighting noise approximation similar to original code:
    image_array: HxWx3 in [0..255]
    """
    img = image_array.astype(np.float32) / 255.0
    orig_shape = img.shape
    flat = img.reshape(-1, 3)
    cov = np.cov(flat, rowvar=False)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        noise = np.random.randn(3) * lighting_std
        delta = eigvecs.dot(eigvals * noise)
        delta = (delta * 255.0).reshape(1, 1, 3)
        img = img + delta
    except Exception:
        # fallback: add small gaussian noise
        img = img + np.random.randn(*img.shape) * (lighting_std * 0.05)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def random_rotation_cv2(img: np.ndarray, max_deg=10):
    deg = np.random.uniform(-max_deg, max_deg)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if img.ndim == 2:
        rotated = rotated
    return rotated

def random_crop_and_resize_cv2(image_array: np.ndarray, crop_frac=0.9, out_size=(64,64)):
    h, w = image_array.shape[:2]
    ch = int(h * crop_frac)
    cw = int(w * crop_frac)
    if ch >= h or cw >= w:
        return cv2.resize(image_array, (out_size[1], out_size[0]), interpolation=cv2.INTER_AREA)
    top = np.random.randint(0, h - ch + 1)
    left = np.random.randint(0, w - cw + 1)
    crop = image_array[top:top+ch, left:left+cw]
    return cv2.resize(crop, (out_size[1], out_size[0]), interpolation=cv2.INTER_AREA)

# -----------------------
# ImageGenerator (full feature)
# -----------------------
class ImageGenerator:
    def __init__(self, ground_truth_data: Dict[str,int], batch_size: int, image_size: Tuple[int,int],
                 train_keys: List[str], validation_keys: List[str],
                 path_prefix: str = '', saturation_var=0.5, brightness_var=0.5,
                 contrast_var=0.5, lighting_std=0.5, horizontal_flip_probability=0.5,
                 vertical_flip_probability=0.0, do_random_crop: bool = False, grayscale: bool = True):
        self.ground_truth_data = ground_truth_data
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_keys = train_keys
        self.validation_keys = validation_keys
        self.path_prefix = path_prefix or ''
        self.saturation_var = saturation_var
        self.brightness_var = brightness_var
        self.contrast_var = contrast_var
        self.lighting_std = lighting_std
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability
        self.do_random_crop = do_random_crop
        self.grayscale = grayscale
        self._on_epoch_end()

    def _on_epoch_end(self):
        random.shuffle(self.train_keys)
        self.train_index = 0
        self.val_index = 0

    def _resolve_path(self, p: str) -> str:
        if os.path.isfile(p):
            return p
        # try join with prefix
        cand = os.path.join(self.path_prefix, p)
        if os.path.isfile(cand):
            return cand
        # try basename with prefix
        cand2 = os.path.join(self.path_prefix, os.path.basename(p))
        if os.path.isfile(cand2):
            return cand2
        # fallback: return original
        return p

    def _load_image_bgr(self, path: str):
        path = self._resolve_path(path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            # return black image
            img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
        return img

    def _load_image_gray(self, path: str):
        path = self._resolve_path(path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
        return img

    def _augment_rgb(self, img_rgb: np.ndarray) -> np.ndarray:
        # color jitter
        img = color_jitter(img_rgb, self.saturation_var, self.brightness_var, self.contrast_var)
        # lighting noise
        if self.lighting_std:
            img = add_lighting_noise(img, self.lighting_std)
        # random horizontal flip
        if np.random.random() < self.horizontal_flip_probability:
            img = img[:, ::-1]
        # random vertical flip (rare)
        if self.vertical_flip_probability and np.random.random() < self.vertical_flip_probability:
            img = img[::-1, :]
        # rotation
        img = random_rotation_cv2(img, max_deg=10)
        # random crop
        if self.do_random_crop:
            img = random_crop_and_resize_cv2(img, crop_frac=0.9, out_size=(self.image_size[0], self.image_size[1]))
        return img

    def _augment_gray(self, img_gray: np.ndarray) -> np.ndarray:
        # convert gray->rgb for color_jitter/lighting only if needed (we'll fake 3 channels)
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        img_rgb = self._augment_rgb(img_rgb)
        # convert back to gray
        img_gray_new = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        return img_gray_new

    def _batch_from_keys(self, keys_slice: List[str], augment: bool):
        X = np.zeros((len(keys_slice), self.image_size[0], self.image_size[1], 1 if self.grayscale else 3), dtype=np.float32)
        y = np.zeros((len(keys_slice),), dtype=np.int32)
        for i, k in enumerate(keys_slice):
            if self.grayscale:
                img = self._load_image_gray(k)
                if augment:
                    img = self._augment_gray(img)
                img = preprocess_input(img.astype(np.float32))
                img = np.expand_dims(img, axis=-1)
            else:
                img = self._load_image_bgr(k)  # BGR
                if augment:
                    img = self._augment_rgb(img)
                # convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = preprocess_input(img.astype(np.float32))
            X[i] = img
            y[i] = int(self.ground_truth_data.get(k, 0))
        y_cat = to_categorical(y, num_classes=num_classes)
        return X, y_cat

    def flow(self, mode='train'):
        while True:
            if mode == 'train':
                keys = self.train_keys
                # shuffle each epoch
                random.shuffle(keys)
                i = 0
                while True:
                    start = i
                    end = i + self.batch_size
                    if end <= len(keys):
                        batch_keys = keys[start:end]
                        i = end
                    else:
                        # wrap around
                        batch_keys = keys[start:len(keys)] + keys[0:(end - len(keys))]
                        i = end - len(keys)
                    X, y = self._batch_from_keys(batch_keys, augment=True)
                    yield X, y
            elif mode in ('val', 'validation'):
                keys = self.validation_keys
                j = 0
                while True:
                    start = j
                    end = j + self.batch_size
                    if end <= len(keys):
                        batch_keys = keys[start:end]
                        j = end
                    else:
                        batch_keys = keys[start:len(keys)]
                        # pad with last to keep batch size
                        while len(batch_keys) < self.batch_size:
                            batch_keys.append(keys[-1])
                        j = len(keys)
                    X, y = self._batch_from_keys(batch_keys, augment=False)
                    yield X, y
            else:
                raise ValueError('Unknown mode for flow(): %s' % mode)

# -----------------------
# Model: mini_XCEPTION (tf.keras)
# -----------------------
def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)
    img_input = Input(shape=input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
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

# -----------------------
# MAIN training
# -----------------------
def main():
    # 1) load mapping
    gt = load_imdb_mat(mat_path, images_path_prefix=images_path)
    print('Total valid samples after filtering:', len(gt))
    if len(gt) == 0:
        raise RuntimeError('No samples found. Check mat_path / images_path prefix / file existence.')

    # 2) split
    train_keys, val_keys = split_imdb_data(gt, validation_split=validation_split, seed=123)
    print('Train samples:', len(train_keys))
    print('Val samples:', len(val_keys))

    # 3) generator
    image_generator = ImageGenerator(
        ground_truth_data=gt,
        batch_size=batch_size,
        image_size=(input_shape[0], input_shape[1]),
        train_keys=train_keys,
        validation_keys=val_keys,
        path_prefix=images_path,
        saturation_var=0.5,
        brightness_var=0.5,
        contrast_var=0.5,
        lighting_std=0.5,
        horizontal_flip_probability=0.5,
        vertical_flip_probability=0.0,
        do_random_crop=do_random_crop,
        grayscale=grayscale
    )

    steps_per_epoch = max(1, int(math.ceil(len(train_keys) / batch_size)))
    validation_steps = max(1, int(math.ceil(len(val_keys) / batch_size)))
    print('Steps per epoch:', steps_per_epoch)
    print('Validation steps:', validation_steps)

    # 4) model
    model = mini_XCEPTION(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # 5) callbacks
    log_file_path = os.path.join(base_path, 'gender_training.log')
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=max(1, int(patience/2)), verbose=1)
    trained_models_path = os.path.join(base_path, 'gender_mini_XCEPTION')
    model_names = f'{trained_models_path}.{{epoch:02d}}-{{val_accuracy:.2f}}.h5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    # 6) train using generator
    print('Start training...')
    start_time = time.time()
    model.fit(
        image_generator.flow(mode='train'),
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=image_generator.flow(mode='val'),
        validation_steps=validation_steps
    )
    elapsed = time.time() - start_time
    print(f'Training completed in {elapsed/60:.2f} minutes.')

if __name__ == '__main__':
    main()
