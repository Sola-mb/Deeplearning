"""
full_comparison.py - Run all comparison experiments for the report
Author: Hadil (Model Builder)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import os
import json
import time

# ==================== CONFIGURATION ====================
MAX_LEN = 150
EMBEDDING_DIM = 100
EPOCHS_COMPARE = 5          # enough to see differences
BATCH_SIZE = 64
USE_SUBSET = True           # use subset for faster comparison
TRAIN_SUBSET = 20000
VAL_SUBSET = 5000

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer', 'tokenizer.json')

# ==================== LOAD FULL DATA ====================
print("Loading data...")
X_train_full = np.load(os.path.join(DATA_DIR, 'X_train_pad.npy'))
y_train_full = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
X_val_full = np.load(os.path.join(DATA_DIR, 'X_val_pad.npy'))
y_val_full = np.load(os.path.join(DATA_DIR, 'y_val.npy'))

# Use subset if needed
if USE_SUBSET:
    np.random.seed(42)
    idx_train = np.random.choice(len(X_train_full), TRAIN_SUBSET, replace=False)
    idx_val = np.random.choice(len(X_val_full), VAL_SUBSET, replace=False)
    X_train = X_train_full[idx_train]
    y_train = y_train_full[idx_train]
    X_val = X_val_full[idx_val]
    y_val = y_val_full[idx_val]
    print(f"Subset used: train {X_train.shape[0]}, val {X_val.shape[0]}")
else:
    X_train, y_train = X_train_full, y_train_full
    X_val, y_val = X_val_full, y_val_full

# Load tokenizer to get vocab size
with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
    tokenizer_json = f.read()
tokenizer = tokenizer_from_json(tokenizer_json)
vocab_size = 20000 + 1   # top 20k + padding

print(f"Vocab size: {vocab_size}")

# ==================== MODEL BUILDERS ====================
def build_single_kernel_cnn(kernel_size=5, dropout_rate=0.5):
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(128, kernel_size, activation='relu', padding='same'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_multikernel_cnn(dropout_rate=0.5):
    inputs = Input(shape=(MAX_LEN,))
    embedding = Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN)(inputs)
    conv_3 = Conv1D(128, 3, activation='relu', padding='same')(embedding)
    conv_5 = Conv1D(128, 5, activation='relu', padding='same')(embedding)
    conv_7 = Conv1D(128, 7, activation='relu', padding='same')(embedding)
    concat = Concatenate()([conv_3, conv_5, conv_7])
    pool = GlobalMaxPooling1D()(concat)
    dense = Dense(64, activation='relu')(pool)
    dropout = Dropout(dropout_rate)(dense)
    outputs = Dense(1, activation='sigmoid')(dropout)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_mlp():
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(model_builder, model_name, extra_args=None):
    print(f"\n--- Training {model_name} ---")
    if extra_args:
        model = model_builder(**extra_args)
    else:
        model = model_builder()
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    start = time.time()
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=EPOCHS_COMPARE,
                        batch_size=BATCH_SIZE,
                        callbacks=[early_stop],
                        verbose=0)
    val_acc = max(history.history['val_accuracy'])
    train_time = time.time() - start
    print(f"   Best val accuracy: {val_acc:.4f} (time: {train_time:.1f}s)")
    return val_acc

# ===================== RUN COMPARISONS =====================
results = []

# 1. Single vs Multi kernel
acc_single = train_and_evaluate(build_single_kernel_cnn, "Single-kernel (k=5)", {'kernel_size':5, 'dropout_rate':0.5})
acc_multi0 = train_and_evaluate(build_multikernel_cnn, "Multi-kernel (3,5,7)", {'dropout_rate':0.5})
results.append(("Single-kernel (k=5)", acc_single))
results.append(("Multi-kernel (3,5,7)", acc_multi0))

# 2. Dropout comparison on multi-kernel
for dr in [0.3, 0.5, 0.7]:
    acc = train_and_evaluate(build_multikernel_cnn, f"Multi-kernel (dropout={dr})", {'dropout_rate':dr})
    results.append((f"Multi-kernel (dropout={dr})", acc))

# 3. MLP vs CNN (use multi-kernel as CNN)
acc_mlp = train_and_evaluate(build_mlp, "MLP")
# Already have multi-kernel acc from above, reuse acc_multi0
results.append(("MLP (Flatten+Dense)", acc_mlp))
results.append(("CNN (multi-kernel)", acc_multi0))

# ==================== PRINT SUMMARY TABLE ====================
print("\n" + "="*60)
print("COMPARISON STUDY SUMMARY")
print("="*60)
print(f"{'Model / Configuration':<35} {'Val Accuracy':<15}")
print("-"*60)
unique_results = {}
for name, acc in results:
    if name not in unique_results:
        unique_results[name] = acc
for name, acc in unique_results.items():
    print(f"{name:<35} {acc:.4f}")
print("="*60)

# Note: All experiments used subset data (20k train / 5k val) for speed.
print("\nNote: All comparisons used a subset of 20k train / 5k val samples.")