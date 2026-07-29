import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import SimpleRNN, Input, Dense
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical, plot_model
import time

# ================= 全局設定 =================
TARGET_LINES = 400
BASE_PATH = '/Users/laihao/Desktop/YTY/dataset_80_1010' # ⚠️ 請確認路徑是否正確 (補了底線)

# ================= 1. 資料讀取與前處理函式 =================
def load_sensor_data(folder_path):
    all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    signal_list, label_list = [], []
    
    if len(all_files) == 0:
        print(f"⚠️ 警告：在 {folder_path} 中找不到任何 CSV 檔案！")
        return np.array([]), np.array([])

    for file in all_files:
        filename = os.path.basename(file)
        label = filename.rsplit('_', 1)[0]
        df = pd.read_csv(file)
        features = df[['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']].values

        if features.shape == (TARGET_LINES, 5):
            signal_list.append(features)
            label_list.append(label)
            
    return np.array(signal_list), np.array(label_list)

def load_split_data(base_path):
    print("🔍 正在從三個資料夾獨立載入數據...")
    signal_train_raw, label_train_raw = load_sensor_data(os.path.join(base_path, 'training'))
    signal_val_raw, label_val_raw     = load_sensor_data(os.path.join(base_path, 'validation'))
    signal_test_raw, label_test_raw   = load_sensor_data(os.path.join(base_path, 'test'))

    print(f"✅ 載入完成！ Train: {len(signal_train_raw)}筆 | Val: {len(signal_val_raw)}筆 | Test: {len(signal_test_raw)}筆")
    return (signal_train_raw, label_train_raw), (signal_val_raw, label_val_raw), (signal_test_raw, label_test_raw)

def encode_label(label_train_raw, label_val_raw, label_test_raw):
    encoder = LabelEncoder()
    label_train_encoded = encoder.fit_transform(label_train_raw)
    label_val_encoded   = encoder.transform(label_val_raw)
    label_test_encoded  = encoder.transform(label_test_raw)

    num_classes = len(encoder.classes_)
    y_train = to_categorical(label_train_encoded, num_classes)
    y_val   = to_categorical(label_val_encoded, num_classes)
    y_test  = to_categorical(label_test_encoded, num_classes)

    return y_train, y_val, y_test

def Z_score(signal_train_raw, signal_val_raw, signal_test_raw):
    num_features = 5
    scaler = StandardScaler()

    signal_train_2d = signal_train_raw.reshape(-1, num_features)
    signal_val_2d   = signal_val_raw.reshape(-1, num_features)
    signal_test_2d  = signal_test_raw.reshape(-1, num_features)

    X_train_scaled = scaler.fit_transform(signal_train_2d).reshape(-1, TARGET_LINES, num_features)
    X_val_scaled   = scaler.transform(signal_val_2d).reshape(-1, TARGET_LINES, num_features)
    X_test_scaled  = scaler.transform(signal_test_2d).reshape(-1, TARGET_LINES, num_features)

    return X_train_scaled, X_val_scaled, X_test_scaled

# ================= 主程式：單一模型訓練與繪圖 =================
if __name__ == "__main__":
    (train_raw, y_train_raw), (val_raw, y_val_raw), (test_raw, y_test_raw) = load_split_data(BASE_PATH)
    y_train, y_val, y_test = encode_label(y_train_raw, y_val_raw, y_test_raw)
    X_train_scaled, X_val_scaled, X_test_scaled = Z_score(train_raw, val_raw, test_raw)

    print("\n" + "="*50)
    print("🚀 開始訓練 SimpleRNN(16) 單一模型以繪製學習曲線")
    print("="*50)

    # --- A. 建立 SimpleRNN 模型 (16 顆神經元) ---
    model = keras.Sequential([
        Input(shape=(TARGET_LINES, 5)), 
        SimpleRNN(units=16, return_sequences=False), 
        Dense(3, activation='softmax')
    ])
    # model.summary()

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    custom_adam = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0)
    model.compile(optimizer=custom_adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # --- B. 匯出架構圖 ---
    # try:
    #     plot_model(
    #         model, 
    #         to_file='SimpleRNN16_architecture.png', 
    #         show_shapes=True,      
    #         show_layer_names=True, 
    #         show_layer_activations=True 
    #     )
    #     print("✅ 模型架構圖已成功儲存為 SimpleRNN16_architecture.png")
    # except Exception as e:
    #     print(f"⚠️ 無法生成架構圖 (請確認是否安裝 graphviz): {e}")

    # --- C. 開始訓練 (關閉 EarlyStopping 以觀察完整 500 Epochs) ---
    print("\n⏳ 正在訓練中，請稍候...")
    start_time = time.time()
    
    history = model.fit(
        X_train_scaled, y_train, 
        epochs=500, 
        validation_data=(X_val_scaled, y_val),
        verbose=1 # 開啟進度條讓你觀察
    )
    
    print(f"✅ 訓練完成！耗時: {(time.time() - start_time)/60:.2f} 分鐘")

    # --- D. 最終驗證 ---
    print("\n📝 正在對 Test Set 進行最終驗證...")
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"🏆 最終 Test Accuracy: {accuracy*100:.2f}%\n")

    # --- E. 繪製並儲存圖表 ---
    print("🎨 正在繪製並儲存訓練曲線圖...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # 🌟 修改標題為 SimpleRNN
    fig.suptitle('SimpleRNN_16neuron Training Results', fontsize=18, fontweight='bold', y=1.02)

    color_train = '#C46D4B'  
    color_val = '#5FAAA0'    

    # 左圖：Accuracy
    ax1.plot(history.history['accuracy'], color=color_train, linewidth=2.5, label='Train')
    ax1.plot(history.history['val_accuracy'], color=color_val, linewidth=2.5, label='Val')
    ax1.set_xlabel('Epochs', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.set_title('a) Model Accuracy', fontsize=14, pad=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right')
    ax1.set_ylim([0.0, 1.0])

    # 右圖：Loss
    ax2.plot(history.history['loss'], color=color_train, linewidth=2.5, label='Train')
    ax2.plot(history.history['val_loss'], color=color_val, linewidth=2.5, label='Val')
    ax2.set_xlabel('Epochs', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=14)
    ax2.set_title('b) Model Loss', fontsize=14, pad=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')
    # Loss 預留較大空間，因為 SimpleRNN 沒學好時 Loss 可能會衝很高
    ax2.set_ylim([0.0, 2.0]) 
    
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    # 加上 Test Accuracy 文字方塊
    fig.text(0.5, 0.03, f'Final Test Accuracy: {accuracy*100:.2f}%', 
             ha='center', va='center', fontsize=16, fontweight='bold', color='#333333',
             bbox=dict(facecolor='#F5F5F5', edgecolor='#CCCCCC', boxstyle='round,pad=0.5'))

    # 存檔 (🌟 修改檔名為 SimpleRNN)
    plt.savefig('SimpleRNN_16neuron_Training_Results.png', dpi=300, bbox_inches='tight')
    print("✅ 訓練曲線已成功儲存為 SimpleRNN_16neuron_Training_Results.png！\n")