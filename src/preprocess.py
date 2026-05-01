

import pandas as pd
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class TextPreprocessor:
    def __init__(self, max_words=10000, max_len=100):
        """
        max_words: أكبر عدد من الكلمات التي سيحتفظ بها (الأكثر تكراراً)
        max_len: طول كل مراجعة بعد التسوية (padding)
        """
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words)
    
    def clean_text(self, text):
        """
        تنظيف نص واحد:
        - تحويل إلى حروف صغيرة
        - إزالة علامات الترقيم
        - إزالة الأرقام (اختياري - حسب بياناتكم)
        """
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)  # إزالة . ، ! ? ( )
            text = re.sub(r'\d+', '', text)      # إزالة الأرقام
            text = re.sub(r'\s+', ' ', text)     # إزالة المسافات الزائدة
            return text.strip()
        return ""
    
    def fit_transform(self, texts):
        """
        تدريب tokenizer على النصوص وتحويلها إلى أرقام مع padding
        """
        # تنظيف جميع النصوص
        cleaned_texts = [self.clean_text(t) for t in texts]
        
        # تدريب tokenizer
        self.tokenizer.fit_on_texts(cleaned_texts)
        
        # تحويل إلى تسلسل أرقام
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        
        # تسوية الأطوال (padding)
        padded = pad_sequences(sequences, maxlen=self.max_len, 
                              padding='post', truncating='post')
        
        return padded, cleaned_texts
    
    def transform(self, texts):
        """
        تحويل نصوص جديدة (اختبار/تحقق) بنفس tokenizer المدرب
        """
        cleaned_texts = [self.clean_text(t) for t in texts]
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        padded = pad_sequences(sequences, maxlen=self.max_len,
                              padding='post', truncating='post')
        return padded, cleaned_texts
    
    def save_tokenizer(self, filepath='tokenizer.json'):
        """حفظ tokenizer لاستخدامه لاحقاً"""
        tokenizer_json = self.tokenizer.to_json()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(tokenizer_json)
    
    def load_tokenizer(self, filepath='tokenizer.json'):
        """تحميل tokenizer من ملف"""
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        with open(filepath, 'r', encoding='utf-8') as f:
            tokenizer_json = f.read()
        self.tokenizer = tokenizer_from_json(tokenizer_json)


# ============= دوال التصور (Visualization) =============

def plot_training_history(history, save_path='../visualizations/training_history.png'):
    """
    رسم منحنيات التدريب (الدقة والخسارة)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # منحنى الدقة
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # منحنى الخسارة
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"✅ تم حفظ الرسم في: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path='../visualizations/confusion_matrix.png'):
    """
    رسم مصفوفة الارتباك
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"✅ تم حفظ مصفوفة الارتباك في: {save_path}")


def plot_class_distribution(y_train, y_val, y_test, save_path='../visualizations/class_distribution.png'):
    """
    رسم توزيع الفئات في مجموعات البيانات (للتأكد من توازن البيانات)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, data, name in zip(axes, [y_train, y_val, y_test], ['Training', 'Validation', 'Test']):
        positive = sum(data)
        negative = len(data) - positive
        ax.bar(['Positive', 'Negative'], [positive, negative], color=['green', 'red'])
        ax.set_title(f'{name} Set')
        ax.set_ylabel('Count')
        
        # إضافة الأرقام فوق الأعمدة
        ax.text(0, positive, str(positive), ha='center', va='bottom')
        ax.text(1, negative, str(negative), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def show_misclassified_examples(texts, true_labels, pred_labels, num_examples=5):
    """
    عرض أمثلة من التصنيفات الخاطئة لفهم الأخطاء
    """
    misclassified_idx = [i for i in range(len(true_labels)) if true_labels[i] != pred_labels[i]]
    
    print(f"\n🔍 عدد التصنيفات الخاطئة: {len(misclassified_idx)} من أصل {len(true_labels)}")
    print(f"   نسبة الخطأ: {len(misclassified_idx)/len(true_labels)*100:.2f}%\n")
    
    print("="*60)
    print("أمثلة على تصنيفات خاطئة:")
    print("="*60)
    
    for i in misclassified_idx[:num_examples]:
        print(f"\n📝 النص الأصلي: {texts[i][:200]}...")
        print(f"🎯 التصنيف الحقيقي: {'Positive' if true_labels[i]==1 else 'Negative'}")
        print(f"🤖 التصنيف المتوقع: {'Positive' if pred_labels[i]==1 else 'Negative'}")
        print("-"*60)



# ============= دالة تحميل البيانات المعالجة =============

def load_processed_data(data_path='../data/'):
    """
    تحميل الملفات المعالجة من مجلد data
    استعملي هذا الدالة باش ترجعي البيانات الجاهزة للتدريب
    
    Parameters:
    -----------
    data_path : str
        المسار إلى مجلد data (افتراضي '../data/')
    
    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test, tokenizer, max_len
    """
    import numpy as np
    import json
    import os
    
    # التحقق من وجود الملفات
    required_files = ['X_train_pad.npy', 'X_val_pad.npy', 'X_test_pad.npy',
                      'y_train.npy', 'y_val.npy', 'y_test.npy', 'tokenizer.json']
    
    for f in required_files:
        if not os.path.exists(os.path.join(data_path, f)):
            raise FileNotFoundError(f"❌ الملف {f} غير موجود في {data_path}")
    
    # تحميل المصفوفات
    print("📂 جاري تحميل البيانات المعالجة...")
    X_train = np.load(os.path.join(data_path, 'X_train_pad.npy'))
    X_val = np.load(os.path.join(data_path, 'X_val_pad.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test_pad.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    y_val = np.load(os.path.join(data_path, 'y_val.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))
    
    # تحميل tokenizer
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    with open(os.path.join(data_path, 'tokenizer.json'), 'r', encoding='utf-8') as f:
        tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)
    
    # قراءة المعلومات (اختياري)
    info_path = os.path.join(data_path, 'preprocessing_info.txt')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            print(f.read())
    
    # استخراج max_len من شكل البيانات
    max_len = X_train.shape[1]
    
    print(f"\n✅ تم التحميل بنجاح!")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_val: {X_val.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   max_len = {max_len}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, tokenizer, max_len        