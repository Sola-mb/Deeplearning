"""
show_predictions.py - Display test reviews with model predictions
Author: Hadil
"""

import numpy as np
import tensorflow as tf
import os
import pandas as pd
from tensorflow.keras.preprocessing.text import tokenizer_from_json


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'final_model_real.keras')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer', 'tokenizer.json')
DATA_DIR = os.path.join(BASE_DIR, 'data')


print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)


print("Loading tokenizer...")
with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
    tokenizer_json_string = f.read()
tokenizer = tokenizer_from_json(tokenizer_json_string)


print("Loading test data...")
X_test = np.load(os.path.join(DATA_DIR, 'X_test_pad.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
print(f"Test data shape: {X_test.shape}")
print(f"Positive ratio in test: {y_test.mean():.3f}")

def decode_sequence(sequence, tokenizer):
    """Convert a padded sequence of integers back to text."""
    word_index = tokenizer.word_index
    index_to_word = {idx: word for word, idx in word_index.items()}
    
   
    words = []
    for idx in sequence:
        if idx == 0:  # padding index
            continue
        if idx in index_to_word:
            words.append(index_to_word[idx])
        else:
            words.append('?')  # unknown
    return ' '.join(words)

# ==================== GET PREDICTIONS ====================
print("Getting predictions...")
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

# ==================== DISPLAY SAMPLE RESULTS ====================
# Show first 20 test reviews
num_samples = 20

results = []
for i in range(min(num_samples, len(X_test))):
    original_text = decode_sequence(X_test[i], tokenizer)
    true_label = "Positive" if y_test[i] == 1 else "Negative"
    pred_label = "Positive" if y_pred[i] == 1 else "Negative"
    confidence = y_pred_prob[i][0]
    correct = "✅" if y_test[i] == y_pred[i] else "❌"
    
    results.append({
        'Review #': i+1,
        'Text': original_text[:100] + "..." if len(original_text) > 100 else original_text,
        'True': true_label,
        'Predicted': pred_label,
        'Confidence': f"{confidence:.3f}",
        'Correct': correct
    })

df = pd.DataFrame(results)
print("\n" + "="*80)
print("SAMPLE TEST REVIEWS WITH PREDICTIONS")
print("="*80)
print(df.to_string(index=False))

# ==================== SAVE TO CSV ====================
df.to_csv(os.path.join(BASE_DIR, 'models', 'predictions_sample.csv'), index=False)
print(f"\n✅ Saved sample predictions to models/predictions_sample.csv")

# ==================== OVERALL ACCURACY ON TEST SET ====================
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print(f"\n📊 Overall Test Accuracy: {acc*100:.2f}%")