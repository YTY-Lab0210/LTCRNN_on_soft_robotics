import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile

# 讓終端機保持乾淨，避免 TF 警告訊息洗版
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical

# ================= 1. 全局設定 =================
TARGET_LINES = 400
# 🚨 請確認資料夾路徑是否正確
BASE_PATH = r'C:\Users\HAO\Desktop\YTY_from_macbook\dataset_xx2020\dataset_602020_zscore' 

# 測試的模型架構 (新增 Explicit 與 Hybrid LTC 的對決！)
# model_configs = ['1D-CNN', 'SimpleRNN-8', 'LSTM-8', 'LTC-4 (Explicit)', 'LTC-4 (Hybrid)']
model_configs = ['LTC-4 (Explicit)', 'LTC-4 (Hybrid)']

num_runs = 5
drop_rates_to_test = [0.0, 0.2, 0.4, 0.6, 0.8]

# 儲存成績的字典
all_test_accuracies = {name: {dr: [] for dr in drop_rates_to_test} for name in model_configs}

# ================= 2. 資料讀取函式 =================
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

# ================= 3. 模擬斷訊歸零 (Signal Erasure) =================
def apply_packet_loss(X_batch, drop_rate):
    """
    模擬遠端斷訊 (Signal Erasure)：
    若該時間步掉包，感測器數值直接歸零。
    """
    if drop_rate == 0.0:
        return X_batch.copy()
        
    X_dropped = X_batch.copy()
    batch_size, time_steps, features = X_dropped.shape
    
    for i in range(batch_size):
        for t in range(1, time_steps):
            if np.random.rand() < drop_rate:
                X_dropped[i, t, :] = 0.0 
    return X_dropped

# ================= 4. 定義 LTC 顯式歐拉神經元 (陽春版) =================
class LTCExplicitNeuron(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(LTCExplicitNeuron, self).__init__(**kwargs)
        self.units = units
        self.state_size = units  

    def build(self, input_shape):
        input_dim = input_shape[-1] 
        init_w = keras.initializers.RandomNormal(stddev=0.5) 
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
        
        # 陽春版：顯式歐拉更新 (容易因數值不穩定而過衝)
        dx = -damping * x + driving
        x_new = x + (delta_t * dx)
        
        return x_new, [x_new]

# ================= 5. 定義 LTC 混合歐拉神經元 (無條件穩定進階版) =================
class LTCHybridNeuron(keras.layers.Layer):
    def __init__(self, units, sub_steps=3, **kwargs):
        super(LTCHybridNeuron, self).__init__(**kwargs)
        self.units = units
        self.sub_steps = sub_steps
        self.state_size = units  

    def build(self, input_shape):
        input_dim = input_shape[-1] 
        init_w = keras.initializers.RandomNormal(stddev=0.5) 
        init_ones = 'ones'
        init_zeros = 'zeros'
        self.w = self.add_weight(shape=(input_dim, self.units), initializer=init_w, trainable=True)
        self.r = self.add_weight(shape=(input_dim, self.units), initializer=init_ones, trainable=True)
        self.mu = self.add_weight(shape=(input_dim, self.units), initializer=init_zeros, trainable=True)

    def call(self, inputs, states):
        x = states[0]  
        base_delta_t = 0.01   
        dt = base_delta_t / self.sub_steps # 細化步長
        
        inputs_expanded = tf.expand_dims(inputs, axis=-1)
        sigma = tf.math.sigmoid(inputs_expanded * self.r + self.mu)
        damping = 1.0 + tf.reduce_sum((tf.abs(self.w) * sigma), axis=1)
        driving = tf.reduce_sum((self.w * sigma), axis=1)
        
        x_new = x
        # 🌟 執行 sub_steps 次高精度 Hybrid Euler 更新
        for _ in range(self.sub_steps):
            # Hasani 論文的核心公式：無條件穩定的半隱式半顯式求解
            x_new = (x_new + dt * driving) / (1.0 + dt * damping)
            
        return x_new, [x_new]

# ================= 6. 主程式：抗噪基準測試 =================
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"🚀 開始執行多模型斷訊歸零容錯基準測試 (Solver Ablation Benchmark)")
    print(f"{'='*70}")
    
    # --- A. 載入數據 ---
    X_train_raw, y_train_raw = load_sensor_data(os.path.join(BASE_PATH, 'training'))
    X_val_raw, y_val_raw     = load_sensor_data(os.path.join(BASE_PATH, 'validation'))
    X_test_raw, y_test_raw   = load_sensor_data(os.path.join(BASE_PATH, 'test'))
    
    # --- B. 標籤轉換 ---
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_raw)
    y_val_encoded   = encoder.transform(y_val_raw)
    y_test_encoded  = encoder.transform(y_test_raw)

    num_classes = len(encoder.classes_)
    y_train = to_categorical(y_train_encoded, num_classes)
    y_val   = to_categorical(y_val_encoded, num_classes)
    y_test  = to_categorical(y_test_encoded, num_classes)

    # --- C. 數據標準化 (Z-score) ---
    num_features = 5
    scaler = StandardScaler()
    
    X_train_2d = X_train_raw.reshape(-1, num_features)
    X_val_2d   = X_val_raw.reshape(-1, num_features)
    X_test_2d  = X_test_raw.reshape(-1, num_features)

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(-1, TARGET_LINES, num_features)
    X_val_scaled   = scaler.transform(X_val_2d).reshape(-1, TARGET_LINES, num_features)
    X_test_scaled  = scaler.transform(X_test_2d).reshape(-1, TARGET_LINES, num_features)

    # --- D. 外層迴圈：切換不同的模型架構 ---
    for name in model_configs:
        print(f"\n{'='*40}")
        print(f"🔹 [目前測試架構：{name}]")
        print(f"{'='*40}")

        # --- E. 內層迴圈：重複訓練 10 次 ---
        for run in range(num_runs):
            print(f"  ▶ 開始第 {run + 1:02d}/{num_runs} 次訓練... ", end="", flush=True)
            
            # 動態建立模型
            if name == '1D-CNN':
                model = keras.Sequential([
                    keras.Input(shape=(TARGET_LINES, 5)),
                    keras.layers.Conv1D(filters=8, kernel_size=5, activation='relu'),
                    keras.layers.GlobalAveragePooling1D(),
                    keras.layers.Dense(3, activation='softmax')
                ])
            elif name == 'SimpleRNN-8':
                model = keras.Sequential([
                    keras.Input(shape=(TARGET_LINES, 5)),
                    keras.layers.SimpleRNN(8, return_sequences=False),
                    keras.layers.Dense(3, activation='softmax')
                ])
            elif name == 'LSTM-8':
                model = keras.Sequential([
                    keras.Input(shape=(TARGET_LINES, 5)),
                    keras.layers.LSTM(8, return_sequences=False),
                    keras.layers.Dense(3, activation='softmax')
                ])
            elif name == 'LTC-4 (Explicit)':
                model = keras.Sequential([
                    keras.Input(shape=(TARGET_LINES, 5)),
                    keras.layers.RNN(LTCExplicitNeuron(units=4), return_sequences=False),
                    keras.layers.Dense(3, activation='softmax')
                ])
            elif name == 'LTC-4 (Hybrid)':
                model = keras.Sequential([
                    keras.Input(shape=(TARGET_LINES, 5)),
                    # 🌟 採用 3 步細化 + Hybrid Euler Solver
                    keras.layers.RNN(LTCHybridNeuron(units=4, sub_steps=3), return_sequences=False),
                    keras.layers.Dense(3, activation='softmax')
                ])

            # 編譯模型
            custom_adam = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0)
            model.compile(optimizer=custom_adam, loss='categorical_crossentropy', metrics=['accuracy'], jit_compile=True)

            # 加入 EarlyStopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=30, restore_best_weights=True, verbose=0
            )

            # 訓練
            history = model.fit(
                X_train_scaled, y_train, 
                epochs=500,
                batch_size=8, 
                validation_data=(X_val_scaled, y_val),
                callbacks=[early_stopping],
                verbose=0 
            )
            print(f"訓練完成 (最佳 Val Acc: {max(history.history['val_accuracy'])*100:.1f}%)")

            # --- F. 殘酷盲測：針對不同掉包率進行測試 ---
            for dr in drop_rates_to_test:
                X_test_dropped = apply_packet_loss(X_test_scaled, dr)
                loss, accuracy = model.evaluate(X_test_dropped, y_test, verbose=0)
                all_test_accuracies[name][dr].append(accuracy * 100)

        # 印出該模型在不同掉包率下的 10 次平均成績
        print(f"\n📊 {name} 10次平均容錯成績 (Signal Erasure):")
        for dr in drop_rates_to_test:
            avg_acc = np.mean(all_test_accuracies[name][dr])
            print(f"    - 掉包率 {dr*100:2.0f}%: {avg_acc:.2f}%")

    # ================= 7. 整理數據並存檔 =================
    records = []
    for name in model_configs:
        for dr in drop_rates_to_test:
            for run in range(num_runs):
                records.append({
                    "Model": name,
                    "Drop_Rate": dr * 100,
                    "Run": run + 1,
                    "Test_Accuracy": all_test_accuracies[name][dr][run]
                })
    df_results = pd.DataFrame(records)
    df_results.to_csv("Benchmark_PacketLoss_Results.csv", index=False)
    print("\n✅ 所有結果已成功儲存至 Benchmark_PacketLoss_Results.csv")

    # ================= 8. 繪製抗噪折線圖 (Line Plot) =================
    plt.figure(figsize=(10, 6), dpi=300)
    
    styles = {
        '1D-CNN': {'color': '#1f77b4', 'marker': 's'},
        'SimpleRNN-8': {'color': '#ff7f0e', 'marker': '^'},
        'LSTM-8': {'color': '#2ca02c', 'marker': 'D'},
        'LTC-4 (Explicit)': {'color': '#7f8c8d', 'marker': 'x'}, # 陽春版 (灰色)
        'LTC-4 (Hybrid)': {'color': '#d62728', 'marker': 'o'}     # 混合歐拉版 (紅色亮點！)
    }

    for name in model_configs:
        mean_accs = [np.mean(all_test_accuracies[name][dr]) for dr in drop_rates_to_test]
        std_accs = [np.std(all_test_accuracies[name][dr]) for dr in drop_rates_to_test]
        
        plt.plot([dr * 100 for dr in drop_rates_to_test], mean_accs, 
                 label=name, color=styles[name]['color'], marker=styles[name]['marker'], 
                 linewidth=2.5, markersize=8)
        
        plt.fill_between([dr * 100 for dr in drop_rates_to_test], 
                         np.array(mean_accs) - np.array(std_accs), 
                         np.array(mean_accs) + np.array(std_accs), 
                         color=styles[name]['color'], alpha=0.15)

    plt.title("Robustness Comparison: Explicit vs. Hybrid Solver (10 Runs Average)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Packet Drop Rate (Signal Erasure) [%]", fontsize=14, fontweight='bold')
    plt.ylabel("Test Accuracy [%]", fontsize=14, fontweight='bold')
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='lower left')
    
    plt.tight_layout()
    plt.savefig("Benchmark_PacketLoss_LinePlot.png", dpi=300, bbox_inches='tight')
    print("📊 完美！結合 Hybrid Solver 對比的折線圖已成功儲存為 Benchmark_PacketLoss_LinePlot.png")
    
    plt.show()