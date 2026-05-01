import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, DepthwiseConv2D, SeparableConv2D,
    BatchNormalization, Activation,
    AveragePooling2D, Dropout, Flatten, Dense, Input
)
from tensorflow.keras.models import Model

LABEL_MAP = {
    0: "Yes",
    1: "No",
    2: "Help",
    3: "Pain",
    4: "Water",
    5: "Toilet",
    6: "Doctor",
    7: "Family"
}


def build_eegnet(nb_classes=8, n_channels=64, n_samples=256, dropout=0.5):
    inp = Input(shape=(n_channels, n_samples, 1))

    # Block 1 — temporal + spatial
    x = Conv2D(8, (1, 64), padding='same', use_bias=False)(inp)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D(
            (n_channels, 1),
            use_bias=False,
            depth_multiplier=2,
            depthwise_constraint=tf.keras.constraints.max_norm(1.)
        )(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(dropout)(x)

    # Block 2 — separable conv
    x = SeparableConv2D(16, (1, 16), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(dropout)(x)

    x   = Flatten()(x)
    out = Dense(nb_classes, activation='softmax')(x)

    return Model(inputs=inp, outputs=out)


def load_model(path='eeg_model.h5'):
    try:
        model = tf.keras.models.load_model(path)
        print(f"✅ Model loaded from {path}")
        return model
    except Exception as e:
        print(f"⚠️  No saved model found. Building fresh EEGNet. ({e})")
        model = build_eegnet()
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


def preprocess_signal(raw_data: np.ndarray,
                      n_channels=64, n_samples=256) -> np.ndarray:
    # Fix channels
    if raw_data.shape[0] < n_channels:
        pad = np.zeros((n_channels - raw_data.shape[0], raw_data.shape[1]))
        raw_data = np.vstack([raw_data, pad])
    else:
        raw_data = raw_data[:n_channels, :]

    # Fix samples
    if raw_data.shape[1] < n_samples:
        pad = np.zeros((n_channels, n_samples - raw_data.shape[1]))
        raw_data = np.hstack([raw_data, pad])
    else:
        raw_data = raw_data[:, :n_samples]

    # Z-score normalize per channel
    mean     = raw_data.mean(axis=1, keepdims=True)
    std      = raw_data.std(axis=1,  keepdims=True) + 1e-8
    raw_data = (raw_data - mean) / std

    return raw_data.reshape(1, n_channels, n_samples, 1).astype(np.float32)


def predict(model, X: np.ndarray):
    probs    = model.predict(X, verbose=0)[0]
    class_id = int(np.argmax(probs))
    return {
        "word":             LABEL_MAP[class_id],
        "confidence":       round(float(np.max(probs)) * 100, 1),
        "all_probabilities": {
            LABEL_MAP[i]: round(float(p) * 100, 1)
            for i, p in enumerate(probs)
        }
    }