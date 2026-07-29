import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Input
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping # 🌟 1. 匯入 EarlyStopping 模組
import matplotlib.pyplot as plt

TARGET_LINES = 400

# ================= 1. 定義讀取資料夾的專屬函式 =================
def load_sensor_data(folder_path):
    """從指定資料夾讀取所有 CSV 並回傳特徵 X 與標籤 y"""
    all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    X_list = []
    y_list = []
    
    if len(all_files) == 0:
        print(f"⚠️ 警告：在 {folder_path} 中找不到任何 CSV 檔案！")
        return np.array([]), np.array([])

    for file in all_files:
        filename = os.path.basename(file)
        label = filename.rsplit('_', 1)[0]

        df = pd.read_csv(file)
        features = df[['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']].values

        if features.shape == (TARGET_LINES, 5):
            X_list.append(features)
            y_list.append(label)
            
    return np.array(X_list), np.array(y_list)

# ================= 2. 分別載入三個資料夾的數據 =================
base_path = '/Users/laihao/Desktop/YTY/dataset_5_1010_zscore' # 確保這裡是你的新資料夾路徑

print("🔍 正在從三個資料夾獨立載入數據...")
X_train_raw, y_train_raw = load_sensor_data(os.path.join(base_path, 'training'))
X_val_raw, y_val_raw     = load_sensor_data(os.path.join(base_path, 'validation'))
X_test_raw, y_test_raw   = load_sensor_data(os.path.join(base_path, 'test'))

print(f"✅ 載入完成！ Train: {len(X_train_raw)}筆 | Val: {len(X_val_raw)}筆 | Test: {len(X_test_raw)}筆")

# ================= 3. 標籤轉換 (Label Encoding) =================
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train_raw)
y_val_encoded   = encoder.transform(y_val_raw)
y_test_encoded  = encoder.transform(y_test_raw)

num_classes = len(encoder.classes_)
y_train = to_categorical(y_train_encoded, num_classes)
y_val   = to_categorical(y_val_encoded, num_classes)
y_test  = to_categorical(y_test_encoded, num_classes)

# ================= 4. 數據標準化 =================
# num_features = 5
# scaler = StandardScaler()

# X_train_2d = X_train_raw.reshape(-1, num_features)
# X_val_2d   = X_val_raw.reshape(-1, num_features)
# X_test_2d  = X_test_raw.reshape(-1, num_features)

# X_train_scaled = scaler.fit_transform(X_train_2d).reshape(-1, TARGET_LINES, num_features)
# X_val_scaled   = scaler.transform(X_val_2d).reshape(-1, TARGET_LINES, num_features)
# X_test_scaled  = scaler.transform(X_test_2d).reshape(-1, TARGET_LINES, num_features)

# ================= 5. 建立 1D-CNN 架構 =================
model = Sequential()
model.add(Input(shape=(TARGET_LINES, 5)))
model.add(Conv1D(filters=15, kernel_size=3, strides=2, activation='relu'))
model.add(Conv1D(filters=15, kernel_size=3, strides=2, activation='relu'))
model.add(Conv1D(filters=15, kernel_size=3, strides=2, activation='relu'))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ================= 6. 開始訓練 =================
print("\n🔥 開始訓練 (掛載 Early Stopping 提早交卷機制)...")

# 🌟 2. 設定 EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# 🌟 3. 將 epochs 調大，並把 early_stop 放進 callbacks 清單
history = model.fit(
    X_train_raw, y_train, 
    epochs=100,             
    batch_size=8, 
    validation_data=(X_val_raw, y_val),
    callbacks=[early_stop]  # 裝上煞車系統
)

# ================= 7. 最終期末考驗證 =================
print("\n📝 正在對 Test Set 進行最終驗證...")
loss, accuracy = model.evaluate(X_test_raw, y_test, verbose=0)
print(f"🏆 最終 Test Accuracy: {accuracy*100:.2f}%\n")


# ================= 8. 繪製訓練收斂曲線 =================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
fig.suptitle('1D-CNN (Stride=2) Training Results', fontsize=18, fontweight='bold', y=1.02)

color_train = '#C46D4B'  
color_val = '#5FAAA0'    

# ----------------- 左圖：Accuracy -----------------
ax1.plot(history.history['accuracy'], color=color_train, linewidth=2.5)
ax1.plot(history.history['val_accuracy'], color=color_val, linewidth=2.5)
ax1.text(len(history.history['accuracy'])-2, history.history['accuracy'][-1]-0.05, 'Train', color=color_train, fontsize=12, fontweight='bold')
ax1.text(len(history.history['val_accuracy'])-2, history.history['val_accuracy'][-1]+0.02, 'Val', color=color_val, fontsize=12, fontweight='bold')
ax1.set_xlabel('Epochs', fontsize=14)
ax1.set_ylabel('Accuracy', fontsize=14)
ax1.set_title('a) Model Accuracy', fontsize=14, pad=10)
ax1.grid(True, linestyle='--', alpha=0.5)

# ----------------- 右圖：Loss -----------------
ax2.plot(history.history['loss'], color=color_train, linewidth=2.5)
ax2.plot(history.history['val_loss'], color=color_val, linewidth=2.5)
ax2.text(len(history.history['loss'])-2, history.history['loss'][-1]+0.05, 'Train', color=color_train, fontsize=12, fontweight='bold')
ax2.text(len(history.history['val_loss'])-2, history.history['val_loss'][-1]+0.05, 'Val', color=color_val, fontsize=12, fontweight='bold')
ax2.set_xlabel('Epochs', fontsize=14)
ax2.set_ylabel('Loss', fontsize=14)
ax2.set_title('b) Model Loss', fontsize=14, pad=10)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('1dcnn_curves_5data_earlystop.png', dpi=300, bbox_inches='tight')
print("\n📊 訓練曲線已儲存為 1dcnn_curves_1data_earlystop.png")