import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
import tempfile
import seaborn as sns  # 🌟 新增這行：強大的統計視覺化套件
from dataset_loader_3ch import DEFAULT_DATASET_PATH, load_split_dataset, zscore_from_train

# 讓終端機保持乾淨，消除不必要的警告
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ================= 1. 全局設定 =================
TARGET_LINES = 400
BASE_PATH = DEFAULT_DATASET_PATH

# 你的參賽選手
model_configs = ['1D-CNN', 'SimpleRNN-8', 'LSTM-8', 'LTC-4']

# 💡 少樣本設定：每種手勢類別抽取 N 筆資料
few_shot_sizes = [60, 30, 15, 6, 3] 
num_runs = 10 # ⚠️ 測試階段可先改為 2，正式跑再改回 10

# 儲存結果的超大字典 { 'LTC-4': {60: [], 30: [], ...}, ... }
all_results = {model: {size: [] for size in few_shot_sizes} for model in model_configs}

# ================= 2. 資料讀取函式 =================
def load_sensor_data(folder_path):
    all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    signal_list, label_list = [], []
    for file in all_files:
        filename = os.path.basename(file)
        label = filename.rsplit('_', 1)[0]
        df = pd.read_csv(file)
        features = df[['Thumb', 'Middle', 'Pinky']].values
        if features.shape == (TARGET_LINES, 3):
            signal_list.append(features)
            label_list.append(label)
    return np.array(signal_list), np.array(label_list)

# 💡 隨機抽樣函式：從完整 Training Set 隨機抽出 N 筆/每類別
def get_few_shot_subset(X, y_raw, samples_per_class):
    unique_classes = np.unique(y_raw)
    selected_indices = []
    
    for cls in unique_classes:
        cls_idx = np.where(y_raw == cls)[0]
        # 如果原始資料比你要抽的還少，就全拿；否則隨機抽
        if len(cls_idx) <= samples_per_class:
            selected_indices.extend(cls_idx)
        else:
            selected = np.random.choice(cls_idx, samples_per_class, replace=False)
            selected_indices.extend(selected)
            
    # 打亂順序
    np.random.shuffle(selected_indices)
    return X[selected_indices], y_raw[selected_indices]

# ================= 3. 定義 LTC 神經元 =================
class LTCNeuron(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(LTCNeuron, self).__init__(**kwargs)
        self.units = units
        self.state_size = units  

    def build(self, input_shape):
        input_dim = input_shape[-1] 
        init_w = keras.initializers.RandomNormal(stddev=1.0) 
        init_ones = 'ones'
        init_zeros = 'zeros'

        self.w = self.add_weight(shape=(input_dim, self.units), initializer=init_w, trainable=True)
        self.r = self.add_weight(shape=(input_dim, self.units), initializer=init_ones, trainable=True)
        self.mu = self.add_weight(shape=(input_dim, self.units), initializer=init_zeros, trainable=True)

    def call(self, inputs, states):
        x = states[0]  
        delta_t = 0.01   
        
        inputs_expanded = tf.expand_dims(inputs, axis=-1)
        sigma = tf.math.sigmoid(inputs_expanded * self.r + self.mu)
        damping = 1.0 + tf.reduce_sum((tf.abs(self.w) * sigma), axis=1)
        driving = tf.reduce_sum((self.w * sigma), axis=1)
        
        dx = -damping * x + driving
        x_new = x + (delta_t * dx)
        
        return x_new, [x_new]

# ================= 4. 主程式：少樣本大亂鬥 =================
if __name__ == "__main__":
    print(f"\n{'='*65}")
    print(f"🌪️ 開始執行：資料效率與少樣本盲測 (Data Efficiency Test)")
    print(f"{'='*65}")
    
    # 載入完整數據 (Val 和 Test 永遠不動)
    (X_train_full, y_train_full_raw), (X_val_raw, y_val_raw), (X_test_raw, y_test_raw) = load_split_dataset(
        BASE_PATH,
        normalize=False,
    )
    
    encoder = LabelEncoder()
    encoder.fit(y_train_full_raw)
    num_classes = len(encoder.classes_)
    num_features = 3 

    print(f"通道有: {num_features}\n")
    print(f"類別有: {num_classes}\n")


    # 外層迴圈 1：切換模型
    for name in model_configs:
        print(f"\n🔹 [目前測試架構：{name}]")
        
        # 外層迴圈 2：切換資料量 (60 -> 30 -> 15 -> 6 -> 3)
        for size in few_shot_sizes:
            print(f"  🔻 訓練資料量：每類別 {size} 筆")
            
            # 內層迴圈：進行 10 次獨立隨機盲測
            for run in range(num_runs):
                print(f"     ▶ Run {run + 1:02d}/{num_runs}...", end=" ", flush=True)
                
                # 🔑 1. 隨機抽取少樣本資料
                X_train_sub_raw, y_train_sub_raw = get_few_shot_subset(X_train_full, y_train_full_raw, size)
                
                # 修正 1：將抽樣出來的標籤轉換為 One-hot 編碼
                y_train_sub = to_categorical(encoder.transform(y_train_sub_raw), num_classes)
                y_val = to_categorical(encoder.transform(y_val_raw), num_classes)
                y_test = to_categorical(encoder.transform(y_test_raw), num_classes)
                X_train_sub, X_val_scaled, X_test_scaled, _, _ = zscore_from_train(
                    X_train_sub_raw,
                    X_val_raw,
                    X_test_raw,
                )

                # 🔑 3. 動態建立模型
                if name == '1D-CNN':
                    model = keras.Sequential([
                        keras.Input(shape=(TARGET_LINES, num_features)),
                        keras.layers.Conv1D(filters=15, kernel_size=3, strides=2, padding='valid', activation='relu'),
                        keras.layers.Conv1D(filters=15, kernel_size=3, strides=2, padding='valid', activation='relu'),
                        keras.layers.Conv1D(filters=15, kernel_size=3, strides=2, padding='valid', activation='relu'),
                        keras.layers.Flatten(),
                        keras.layers.Dense(num_classes, activation='softmax')
                    ]) 
                elif name == 'SimpleRNN-8':
                    model = keras.Sequential([
                        keras.Input(shape=(TARGET_LINES, num_features)),
                        keras.layers.SimpleRNN(8, return_sequences=False),
                        keras.layers.Dense(num_classes, activation='softmax')
                    ])
                elif name == 'LSTM-8':
                    model = keras.Sequential([
                        keras.Input(shape=(TARGET_LINES, num_features)),
                        keras.layers.LSTM(8, return_sequences=False),
                        keras.layers.Dense(num_classes, activation='softmax')
                    ])
                elif name == 'LTC-4':
                    model = keras.Sequential([
                        keras.Input(shape=(TARGET_LINES, num_features)),
                        keras.layers.RNN(LTCNeuron(units=4), return_sequences=False),
                        keras.layers.Dense(num_classes, activation='softmax')
                    ])

                model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0), 
                              loss='categorical_crossentropy', metrics=['accuracy'], jit_compile=True)
                
                early_stopping = keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=200, restore_best_weights=True, verbose=0
                )

                # 修正 2：當資料極少時，動態縮小 batch_size
                current_batch_size = min(8, len(X_train_sub_raw))
                
                # 執行訓練 (注意：Y 放的是 y_train_sub)
                history = model.fit(
                    X_train_sub, y_train_sub,
                    epochs=2000,
                    batch_size=current_batch_size, 
                    validation_data=(X_val_scaled, y_val),
                    callbacks=[early_stopping],
                    verbose=0 
                )
                
                # 最終評估
                loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
                
                all_results[name][size].append(accuracy * 100)
                print(f"Acc: {accuracy*100:>5.2f}%")

    print("\n✅ 所有訓練完畢！開始整理數據與繪製箱型圖...")

    # ================= 5. 數據轉換與箱型圖 (Boxplot) 繪製 =================
    
    # 先將大字典攤平，轉換成 Seaborn 喜歡的 DataFrame 格式
    flat_results = []
    for name in model_configs:
        for size in few_shot_sizes:
            for i, acc in enumerate(all_results[name][size]):
                flat_results.append({
                    'Model': name, 
                    'Samples_Per_Class': size, 
                    'Run': i+1, 
                    'Accuracy': acc
                })
    df_results = pd.DataFrame(flat_results)
    
    # 順手存下原始 CSV，以備不時之需
    df_results.to_csv("Few_Shot_Raw_Data.csv", index=False)
    print("💾 原始測試數據已儲存為 Few_Shot_Raw_Data.csv")

    # 準備畫布
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    # 修正 3：沿用你原本精心挑選的顏色配置 (移除了沒用到的 LTC-4，確保顏色不跑掉)
    custom_palette = {
        '1D-CNN': '#ff7f0e',
        'SimpleRNN-8': '#7f7f7f',
        'LSTM-8': '#d62728',
        'LTC-4': '#1f77b4'
    }

    # 使用 Seaborn 繪製分組箱型圖
    sns.boxplot(
        data=df_results, 
        x='Samples_Per_Class', 
        y='Accuracy', 
        hue='Model',            # 根據模型分組
        palette=custom_palette, 
        order=[60, 30, 15, 6, 3],       # 強制 X 軸按照數量遞減排序
        hue_order=['1D-CNN', 'SimpleRNN-8', 'LSTM-8', 'LTC-4'], # 固定圖例順序
        ax=ax,
        width=0.6,              # 箱子的寬度
        fliersize=5,            # 離群值點的大小
        linewidth=1.2           # 框線粗細
    )

    x_order = [60, 30, 15, 6, 3]
    hue_order = ['1D-CNN', 'SimpleRNN-8', 'LSTM-8', 'LTC-4']
    
    # 這裡的 width 必須跟你 sns.boxplot 設定的 width=0.6 一致
    width = 0.6 
    n_hues = len(hue_order)
    # 計算 4 個模型的 X 軸微調偏移量
    offsets = np.linspace(-width/2 + width/(2*n_hues), width/2 - width/(2*n_hues), n_hues)

    for x_idx, size in enumerate(x_order):
        for hue_idx, model in enumerate(hue_order):
            # 抓出這組特地條件的資料
            subset = df_results[(df_results['Samples_Per_Class'] == size) & (df_results['Model'] == model)]
            if not subset.empty:
                mean_val = subset['Accuracy'].mean()
                min_val = subset['Accuracy'].min() # 抓取該箱子的最底端
                x_pos = x_idx + offsets[hue_idx]   # 計算精準的 X 座標
                
                # 在該箱子最低點的下方加上平均值
                ax.text(x_pos, min_val - 1.0, f'{mean_val:.1f}', 
                        ha='center', va='top', fontsize=8, fontweight='bold', color='black',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                        zorder=10)

    # 圖表美化
    ax.set_title("Data Efficiency and Few-Shot Learning Distribution (10 Independent Runs)", 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Number of Training Samples per Class", fontsize=14, fontweight='bold')
    ax.set_ylabel("Test Accuracy (%)", fontsize=14, fontweight='bold')
    
    # 箱型圖通常只需要水平參考線來對齊數值
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', labelsize=12)
    
    # 讓 Y 軸的顯示範圍更合理 (最高 100，最低為整體最差表現減 5)
    min_acc = df_results['Accuracy'].min()
    ax.set_ylim(max(0, min_acc - 5), 100) # 給標籤留點空間

    # 調整圖例位置，避免擋住箱體
    plt.legend(title='Neural Architecture', title_fontsize='12', fontsize='11', 
               loc='lower left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig("Few_Shot_Efficiency_Boxplot.png", dpi=300, bbox_inches='tight')
    print("📊 完美！箱型圖已儲存為 Few_Shot_Efficiency_Boxplot.png")
    
    plt.show()
