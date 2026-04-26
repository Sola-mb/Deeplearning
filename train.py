"""
train.py - Multi-Kernel CNN with Real Amazon Reviews Data
Author: Hadil (Model Builder & Trainer)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import pickle
import os

# ==================== CONFIGURATION ====================
MAX_WORDS = 20000      # from preprocessing_info.txt
MAX_LEN = 150          # from preprocessing_info.txt
EMBEDDING_DIM = 100
BATCH_SIZE = 64
EPOCHS = 10

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer', 'tokenizer.json')

# ==================== LOAD DATA ====================
print("Loading data...")
X_train = np.load(os.path.join(DATA_DIR, 'X_train_pad.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
X_val = np.load(os.path.join(DATA_DIR, 'X_val_pad.npy'))
y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
X_test = np.load(os.path.join(DATA_DIR, 'X_test_pad.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Labels - Train positive ratio: {y_train.mean():.3f}")

# ==================== LOAD TOKENIZER ====================
print(f"Loading tokenizer from {TOKENIZER_PATH}")
with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
    tokenizer_json_string = f.read()   # Read as string, not dict
tokenizer = tokenizer_from_json(tokenizer_json_string)

# Use top MAX_WORDS words (vocab size includes padding index 0)
vocab_size = MAX_WORDS + 1   # +1 for padding index (0)
print(f"Using vocab_size: {vocab_size}")

# ==================== BUILD MULTI-KERNEL CNN ====================
def build_multikernel_cnn(vocab_size, max_len, embedding_dim=100):
    inputs = Input(shape=(max_len,))
    
    embedding = Embedding(vocab_size, embedding_dim, input_length=max_len)(inputs)
    
    # Three parallel convolutions with different kernel sizes
    conv_3 = Conv1D(128, 3, activation='relu', padding='same')(embedding)
    conv_5 = Conv1D(128, 5, activation='relu', padding='same')(embedding)
    conv_7 = Conv1D(128, 7, activation='relu', padding='same')(embedding)
    
    concat = Concatenate()([conv_3, conv_5, conv_7])
    pool = GlobalMaxPooling1D()(concat)
    dense = Dense(64, activation='relu')(pool)
    dropout = Dropout(0.5)(dense)
    outputs = Dense(1, activation='sigmoid')(dropout)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

print("Building multi-kernel CNN...")
model = build_multikernel_cnn(vocab_size, MAX_LEN, EMBEDDING_DIM)
model.summary()

# ==================== CALLBACKS ====================
os.makedirs(MODELS_DIR, exist_ok=True)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)
checkpoint = ModelCheckpoint(
    os.path.join(MODELS_DIR, 'best_model_real.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ==================== TRAINING ====================
print("Starting training on real data...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# ==================== SAVE FINAL MODEL AND HISTORY ====================
model.save(os.path.join(MODELS_DIR, 'final_model_real.keras'))
with open(os.path.join(MODELS_DIR, 'history_real.pkl'), 'wb') as f:
    pickle.dump(history.history, f)

# ==================== EVALUATE ON TEST SET ====================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")
print(f"✅ Test Loss: {test_loss:.4f}")

# Save test accuracy
with open(os.path.join(MODELS_DIR, 'test_accuracy.txt'), 'w') as f:
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")

print("Training complete. Model saved in 'models/' folder.")