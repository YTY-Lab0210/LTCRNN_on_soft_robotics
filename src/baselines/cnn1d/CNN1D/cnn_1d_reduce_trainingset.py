import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Input
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

TARGET_LINES = 400

# ================= 1. 定義讀取資料夾的專屬函式 =================
def load_sensor_data(folder_path):
    """從指定資料夾讀取所有 CSV 並回傳特徵 X 與標籤 y"""
    all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    X_list = []
    y_list = []
    
    if len(all_files) == 0:
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

# ================= 2. 實驗全域設定 =================
base_path = '/Users/laihao/Desktop/YTY' 
dataset_folders = ['dataset_10_1010', 'dataset_20_1010', 'dataset_40_1010', 'dataset_80_1010']
# plot_labels = ['60% Train', '30% Train', '15% Train', '6% Train']
num_runs = 10

# 準備一個二維陣列來儲存所有實驗的準確率
all_accuracies = [] 

# ================= 3. 外層迴圈：切換不同的資料集 =================
for folder_name in dataset_folders:
    print(f"\n{'='*50}")
    print(f"🚀 正在處理資料集: {folder_name}")
    print(f"{'='*50}")
    
    current_dataset_path = os.path.join(base_path, folder_name)
    
    # --- A. 載入數據 ---
    X_train_raw, y_train_raw = load_sensor_data(os.path.join(current_dataset_path, 'training'))
    X_val_raw, y_val_raw     = load_sensor_data(os.path.join(current_dataset_path, 'validation'))
    X_test_raw, y_test_raw   = load_sensor_data(os.path.join(current_dataset_path, 'test'))
    
    print(f"✅ 載入完成 | Train: {len(X_train_raw)}筆 | Val: {len(X_val_raw)}筆 | Test: {len(X_test_raw)}筆")

    # --- B. 標籤轉換 ---
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_raw)
    y_val_encoded   = encoder.transform(y_val_raw)
    y_test_encoded  = encoder.transform(y_test_raw)

    num_classes = len(encoder.classes_)
    y_train = to_categorical(y_train_encoded, num_classes)
    y_val   = to_categorical(y_val_encoded, num_classes)
    y_test  = to_categorical(y_test_encoded, num_classes)

    # --- C. 數據標準化 ---
    num_features = 5
    scaler = StandardScaler()
    
    X_train_2d = X_train_raw.reshape(-1, num_features)
    X_val_2d   = X_val_raw.reshape(-1, num_features)
    X_test_2d  = X_test_raw.reshape(-1, num_features)

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(-1, TARGET_LINES, num_features)
    X_val_scaled   = scaler.transform(X_val_2d).reshape(-1, TARGET_LINES, num_features)
    X_test_scaled  = scaler.transform(X_test_2d).reshape(-1, TARGET_LINES, num_features)

    # 準備一個陣列存放這 10 次的準確率
    current_folder_accuracies = []

    # ================= 4. 內層迴圈：重複訓練 10 次 =================
    for run in range(num_runs):
        print(f"  ▶ 開始第 {run + 1}/{num_runs} 次訓練...", end=" ")
        
        # 🌟 核心關鍵：模型必須在這裡重新建立，確保每次都是全新亂數初始化的權重
        model = Sequential()
        model.add(Input(shape=(TARGET_LINES, 5)))
        model.add(Conv1D(filters=15, kernel_size=3, strides=2, activation='relu'))
        model.add(Conv1D(filters=15, kernel_size=3, strides=2, activation='relu'))
        model.add(Conv1D(filters=15, kernel_size=3, strides=2, activation='relu'))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # 訓練模型 (設定 verbose=0 避免印出大量進度條洗版)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(
            X_train_scaled, y_train, 
            epochs=100,             
            batch_size=8, 
            validation_data=(X_val_scaled, y_val),
            callbacks=[early_stop],
            verbose=0  # 關閉訓練進度條
        )

        # 驗證測試集準確率
        loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
        current_folder_accuracies.append(accuracy)
        print(f"完成！Test Acc: {accuracy*100:.2f}%")
        
    # 將這個資料夾的 10 次結果存入總陣列
    all_accuracies.append(current_folder_accuracies)
    print(f"📊 {folder_name} 平均準確率: {np.mean(current_folder_accuracies)*100:.2f}%")



# ================= 5. 繪製 Boxplot 盒鬚圖 =================
print("\n🎨 正在繪製並儲存 Boxplot 盒鬚圖...")

# 將數據轉換為百分比，方便閱讀
all_accuracies_percent = [[val * 100 for val in exp] for exp in all_accuracies]
plot_labels = ['30', '60', '120', '240']

plt.figure(figsize=(10, 6), dpi=300)

# 畫出盒鬚圖 (加上 zorder=1 確保盒子在底層)
box = plt.boxplot(all_accuracies_percent, tick_labels=plot_labels, patch_artist=True, zorder=1)

# 調整 Boxplot 外觀顏色 (5種顏色以防之後需要5個Box)
colors = ['#C46D4B', '#5FAAA0', '#E5A93D', '#6B5B95', '#D2607D']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 僅保留中位數黑線的樣式，移除原本印數字的程式碼
for median_line in box['medians']:
    median_line.set(color='black', linewidth=2)

# 🌟 新增重點：計算平均值 (Mean)，並把標籤放在該組數據的最底下
for i, data_pts in enumerate(all_accuracies_percent):
    mean_val = np.mean(data_pts)        # 計算 10 次的平均準確率
    min_val = np.min(data_pts)          # 抓出這 10 次裡面的最低分
    
    center_x = i + 1  # Boxplot 的 X 座標預設為 1, 2, 3...
    
    # 將文字放在最低分的下方 (min_val - 2.0)，確保完全不擋到 Box 和資料點
    plt.text(center_x, min_val - 2.0, f'{mean_val:.1f}%', 
             ha='center', va='top', fontsize=11, fontweight='bold', color='black',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#CCCCCC', boxstyle='round,pad=0.3'), zorder=4)

# 疊加個別的 10 次實驗資料點 (Scatter plot with Jitter)
for i, data_pts in enumerate(all_accuracies_percent):
    x_jitter = np.random.normal(i + 1, 0.04, size=len(data_pts))
    plt.scatter(x_jitter, data_pts, color='black', alpha=0.6, s=30, edgecolor='white', linewidth=0.8, zorder=2)

plt.title('1D-CNN Performance Under Different Training Data Ratios (10 Runs/Exp)', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Test Accuracy (%)', fontsize=14)
plt.xlabel('Training Set Size (Number of Samples)', fontsize=14)

plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0) 
plt.ylim(0, 105) 

plt.tight_layout()
plt.savefig('1dcnn_low_data_boxplot.png', dpi=300, bbox_inches='tight')
print("✅ Boxplot 已成功儲存為 1dcnn_low_data_boxplot.png！")