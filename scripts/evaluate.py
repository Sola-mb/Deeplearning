import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'final_model_real.keras')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Load test data
X_test = np.load(os.path.join(DATA_DIR, 'X_test_pad.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Predict
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Multi-Kernel CNN')
plt.savefig(os.path.join(BASE_DIR, 'models', 'confusion_matrix.png'))
print("Confusion matrix saved to models/confusion_matrix.png")