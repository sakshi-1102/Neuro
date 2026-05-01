import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import build_eegnet

print("Generating training data...")
np.random.seed(42)

N, C, S, CLS = 800, 64, 256, 8
X = np.random.randn(N, C, S, 1).astype(np.float32)
y = np.random.randint(0, CLS, N)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42)

print(f"Train:{X_train.shape} Val:{X_val.shape} Test:{X_test.shape}")

model = build_eegnet(nb_classes=CLS, n_channels=C, n_samples=S)
model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

save_path = os.path.join(
    os.path.dirname(__file__), '..', 'backend', 'eeg_model.h5')

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=5, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        save_path, save_best_only=True, verbose=1),
]

print("\nTraining started...")
model.fit(X_train, y_train, epochs=50, batch_size=32,
          validation_data=(X_val, y_val), callbacks=callbacks)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc*100:.1f}%")
model.save(save_path)
print(f"Model saved to {save_path}")