import pickle
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
history_path = os.path.join(BASE_DIR, 'models', 'history_real.pkl')

with open(history_path, 'rb') as f:
    history = pickle.load(f)

plt.figure(figsize=(12,4))


plt.subplot(1,2,1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'models', 'learning_curves.png'))
print("Learning curves saved to models/learning_curves.png")