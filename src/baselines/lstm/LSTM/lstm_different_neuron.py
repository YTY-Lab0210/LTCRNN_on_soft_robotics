import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Input, Dense
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical

import seaborn as sns # 確保有 import
import time # 用來計算執行時間

# ================= 全局設定 =================
TARGET_LINES = 400
BASE_PATH = '/Users/laihao/Desktop/YTY/dataset_80_1010' # 請確認路徑正確

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

# ================= 實驗主程式 =================
if __name__ == "__main__":
    (train_raw, y_train_raw), (val_raw, y_val_raw), (test_raw, y_test_raw) = load_split_data(BASE_PATH)
    y_train, y_val, y_test = encode_label(y_train_raw, y_val_raw, y_test_raw)
    X_train_scaled, X_val_scaled, X_test_scaled = Z_score(train_raw, val_raw, test_raw)

    # 🌟 實驗設計大綱
    unit_configs = [1, 2, 4, 8, 16]
    runs_per_config = 10
    epochs_per_run = 100 # ⚠️ 注意：LSTM 通常不需要跑到 500，且跑太慢，我先幫你調到 100，你可以自行調整
    
    # 用來收集所有結果的清單
    experiment_results = []

    print("\n" + "="*75)
    print(f"🚀 開始 LSTM 基準測試！測試神經元數量: {unit_configs}")
    print(f"📊 每個配置跑 {runs_per_config} 次，每次 {epochs_per_run} Epochs")
    print("="*75)

    global_start_time = time.time()

    # 雙層迴圈：外層換神經元數量，內層跑 10 次
    for u in unit_configs:
        print(f"\n🧠 [目前測試架構：LSTM {u} 顆神經元]")
        
        for r in range(runs_per_config):
            # 🌟 核心修改：換成傳統的 LSTM 層
            model = keras.Sequential([
                Input(shape=(TARGET_LINES, 5)), 
                LSTM(units=u, return_sequences=False), 
                Dense(3, activation='softmax')
            ])

            # 使用標準 Adam 即可 (LSTM 通常不需要調特別大的 learning rate)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # 加上 EarlyStopping 防止過擬合並加速實驗
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

            # 背景靜默訓練
            model.fit(
                X_train_scaled, y_train, 
                epochs=epochs_per_run, 
                validation_data=(X_val_scaled, y_val),
                callbacks=[early_stop],
                verbose=0 
            )

            # 期末考驗證
            loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
            acc_percent = accuracy * 100
            
            # 紀錄資料
            experiment_results.append({
                'Neurons (N)': u,
                'Run': r + 1,
                'Test Accuracy (%)': acc_percent
            })
            
            print(f"   ▶ Run {r+1:02d}/10 完畢 | Test Acc: {acc_percent:>5.2f}%")

    global_end_time = time.time()
    print("\n✅ 所有實驗執行完畢！總耗時: {:.2f} 分鐘".format((global_end_time - global_start_time)/60))

    # ================= 4. 將結果轉為 DataFrame 並繪製神級盒狀圖 =================
    df_results = pd.DataFrame(experiment_results)

    plt.figure(figsize=(10, 6), dpi=300)
    
    # 畫出美觀的漸層色盒狀圖 (改用暖色系以區別 LTC 的圖)
    sns.boxplot(
        x='Neurons (N)', 
        y='Test Accuracy (%)', 
        data=df_results, 
        palette="flare", # 使用暖色漸層帶
        width=0.5, 
        boxprops=dict(alpha=0.7)
    )
    
    # 疊加散點，顯示每一次的真實落點
    sns.stripplot(
        x='Neurons (N)', 
        y='Test Accuracy (%)', 
        data=df_results, 
        color="#333333", 
        size=6, 
        jitter=True, 
        alpha=0.6
    )

    plt.title('Baseline Study: LSTM Capacity vs. Performance (10 Runs)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Number of LSTM Neurons (N)', fontsize=14)
    plt.ylabel('Test Accuracy (%)', fontsize=14)
    
    # 動態設定 Y 軸，確保畫面好看
    y_min = max(0, df_results['Test Accuracy (%)'].min() - 5)
    plt.ylim([y_min, 100])
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('LSTM_Ablation_Boxplot.png', dpi=300, bbox_inches='tight')
    print("\n📊 基準測試盒狀圖已儲存為 LSTM_Ablation_Boxplot.png")