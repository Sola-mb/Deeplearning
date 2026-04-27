import numpy as np
import tensorflow as tf
import json
import re
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'final_model_real.keras')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer', 'tokenizer.json')
MAX_LEN = 150

# Load model and tokenizer
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
    tokenizer_json_string = f.read()
tokenizer = tokenizer_from_json(tokenizer_json_string)
print("Ready!\n")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def predict_review(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='pre', truncating='pre')
    prob = model.predict(padded, verbose=0)[0][0]
    sentiment = "POSITIVE 😊" if prob >= 0.5 else "NEGATIVE 😞"
    return sentiment, prob

# Example reviews
test_reviews = [
    "I absolutely love this product, it's fantastic!",
    "Terrible quality, broke after two days.",
    "Good value for money, works as expected.",
    "Waste of money, do not buy.",
    "Amazing! Best purchase ever."
]

print("Testing on sample reviews:\n")
for rev in test_reviews:
    sent, conf = predict_review(rev)
    print(f"Review: {rev}")
    print(f"Sentiment: {sent} (confidence: {conf:.4f})\n")