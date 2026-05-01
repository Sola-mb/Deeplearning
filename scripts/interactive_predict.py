import numpy as np
import tensorflow as tf
import json
import re
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'final_model_real.keras')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'data', 'tokenizer.json')
MAX_LEN = 150

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
    tokenizer_json_string = f.read()
tokenizer = tokenizer_from_json(tokenizer_json_string)
print("Ready! Type a review (or 'quit')\n")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def predict_review(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='pre', truncating='pre')
    prob = model.predict(padded, verbose=0)[0][0]
    return ("POSITIVE 😊" if prob >= 0.5 else "NEGATIVE 😞"), prob

while True:
    review = input("📝 Enter review: ").strip()
    if review.lower() == 'quit':
        break
    if not review:
        continue
    sent, conf = predict_review(review)
    print(f"→ {sent} (confidence: {conf:.4f})\n")