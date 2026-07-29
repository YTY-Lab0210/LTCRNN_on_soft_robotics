import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# ================= 全局設定 =================
TARGET_LINES = 400
BASE_PATH = r"C:\Users\HAO\Desktop\YTY_from_macbook\dataset_xx2020_new\dataset_602020"
NUM_RUNS = 10  # 每個設定跑 10 次

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
            
    return np.array(signal_list, dtype=np.float32), np.array(label_list)

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

def apply_z_score(train, val, test):
    """
    嚴謹的 Z-score 標準化：
    計算 Training Set 各感測通道的平均值與標準差，並將其套用於 Train, Val, Test，
    避免測試資料的資訊洩漏到前處理階段。
    """
    # 沿著樣本(axis=0)和時間步(axis=1)計算，保留通道(axis=2)的統計量
    mean = np.mean(train, axis=(0, 1))
    std = np.std(train, axis=(0, 1))
    std[std == 0] = 1e-7 # 避免除以零

    train_scaled = (train - mean) / std
    val_scaled = (val - mean) / std
    test_scaled = (test - mean) / std

    return train_scaled, val_scaled, test_scaled

# ================= 2. 定義 4 顆 LTC 神經元 =================
class LTCNeuron(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(LTCNeuron, self).__init__(**kwargs)
        self.units = units
        self.state_size = units  

    def build(self, input_shape):
        input_dim = input_shape[-1] 
        init_w = keras.initializers.RandomNormal(stddev=0.5) 
        init_ones = 'ones'
        init_zeros = 'zeros'

        self.w = self.add_weight(shape=(input_dim, self.units), initializer=init_w, trainable=True, name="w")
        self.r = self.add_weight(shape=(input_dim, self.units), initializer=init_ones, trainable=True, name="r")
        self.mu = self.add_weight(shape=(input_dim, self.units), initializer=init_zeros, trainable=True, name="mu")

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

# ================= 3. 建立模型的工廠函式 =================
def create_compiled_model():
    """每次呼叫都會回傳一個全新初始化的 LTC-4 模型"""
    model = keras.Sequential([
        keras.Input(shape=(TARGET_LINES, 5)), 
        keras.layers.RNN(LTCNeuron(units=8), return_sequences=False), 
        keras.layers.Dense(5, activation='softmax') # 注意：請確定你的類別數是3，如果不是請改回5
    ])
    custom_adam = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0)
    model.compile(optimizer=custom_adam, loss='categorical_crossentropy', metrics=['accuracy'], jit_compile=True)
    return model

# ================= 4. 實驗主流程 =================
def run_experiment(X_train, y_train, X_val, y_val, X_test, y_test, num_runs=10, exp_name=""):
    accuracies = []
    print(f"\n[{exp_name}] 開始進行 {num_runs} 次獨立訓練...")
    
    for i in range(num_runs):
        print(f"  ▶ 正在執行第 {i+1}/{num_runs} 次訓練...", end=" ")
        model = create_compiled_model()
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=50, restore_best_weights=True, verbose=0
        )
        
        # 訓練模型 (設為 verbose=0 避免印出太多進度條)
        model.fit(
            X_train, y_train, 
            epochs=1000,
            batch_size=8, 
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1 
        )
        
        # 評估測試集
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        accuracies.append(accuracy)
        print(f"Test Acc: {accuracy*100:.2f}%")
        
    return accuracies

if __name__ == "__main__":
    # --- A. 準備數據 ---
    (train_raw, y_train_raw), (val_raw, y_val_raw), (test_raw, y_test_raw) = load_split_data(BASE_PATH)
    
    # 確保維度與防呆
    if len(test_raw) == 0:
        raise ValueError("測試集為空！請確認路徑或 CSV 檔案內容是否符合 400 行的設定。")
        
    y_train, y_val, y_test = encode_label(y_train_raw, y_val_raw, y_test_raw)

    # --- B. 執行實驗 (無 Z-score) ---
    print("\n" + "="*50)
    print("實驗一：無 Z-score 前處理 (Raw Data)")
    acc_no_zscore = run_experiment(train_raw, y_train, val_raw, y_val, test_raw, y_test, NUM_RUNS, "Raw Data")

    # --- C. 執行實驗 (有 Z-score) ---
    print("\n" + "="*50)
    print("實驗二：加入 Z-score 前處理")
    train_z, val_z, test_z = apply_z_score(train_raw, val_raw, test_raw)
    acc_with_zscore = run_experiment(train_z, y_train, val_z, y_val, test_z, y_test, NUM_RUNS, "Z-score Data")

   # --- D. 繪製並儲存 Boxplot ---
    print("\n📊 正在繪製消融實驗 Boxplot...")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # 修正 1：必須先轉成 numpy array 才能乘以 100
    data_to_plot = [np.array(acc_no_zscore) * 100, np.array(acc_with_zscore) * 100]
    labels = ['Without Z-score\n(Raw Data)', 'With Z-score\n(Normalized)']
    
    # 繪製箱型圖
    box = ax.boxplot(data_to_plot, patch_artist=True, labels=labels, 
                     widths=0.5, showmeans=True)
    
    # 自訂顏色與外觀
    colors = ['#C46D4B', '#5FAAA0']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for median in box['medians']:
        median.set(color='black', linewidth=2)
    
    # 疊加個別的 10 次實驗資料點 (Scatter plot with Jitter)
    for i, data_pts in enumerate(data_to_plot):
        x_jitter = np.random.normal(i + 1, 0.04, size=len(data_pts))
        plt.scatter(x_jitter, data_pts, color='black', alpha=0.6, s=30, edgecolor='white', linewidth=0.8, zorder=2)

    # 計算平均值，並把標籤放在該組數據的最底下
    for i, data_pts in enumerate(data_to_plot):
        mean_val = np.mean(data_pts)        
        min_val = np.min(data_pts)          
        center_x = i + 1  
        
        # 修正 3：統一把文字畫在這裡，並加入 "Mean:" 字眼
        plt.text(center_x, min_val - 3.0, f'Mean: {mean_val:.1f}%', 
                 ha='center', va='top', fontsize=12, fontweight='bold', color='#333333',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='#CCCCCC', boxstyle='round,pad=0.3'), zorder=4)
    
    # 修正 2：設定標題與軸標籤 (Y 軸改為 0~105 容納百分比)
    ax.set_ylabel('Test Accuracy (%)', fontsize=14)
    ax.set_title('Ablation Study: Effect of Z-score Preprocessing', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylim([-5, 110]) # 上下留一點空間給標籤
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('Zscore_Ablation_Boxplot.png', dpi=300, bbox_inches='tight')
    print("✅ 實驗完成！箱型圖已儲存為 'Zscore_Ablation_Boxplot.png'")