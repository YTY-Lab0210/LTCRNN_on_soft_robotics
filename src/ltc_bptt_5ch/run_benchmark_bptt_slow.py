import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
import seaborn as sns
# ================= 1. 全局設定 =================
TARGET_LINES = 400
BASE_PATH = r'C:\Users\HAO\Desktop\YTY_from_macbook\dataset_xx2020_new\dataset_602020_zscore' 

# 🌟 設定你的測試集資料夾對應
test_folders = {
    '1.0x': 'test',
    '2.0x': 'test_slow_2x',
    '3.0x': 'test_slow_3x'
}

model_configs = ['1D-CNN', 'SimpleRNN-8', 'LSTM-8', 'LTC-8']
num_runs = 10  # 🌟 執行 10 次大亂鬥

# 建立巢狀字典儲存成績：results['1.5x']['LTC-4'] = [85.0, 86.6...]
all_test_accuracies = {factor: {name: [] for name in model_configs} for factor in test_folders.keys()}

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

# ================= 4. 主程式：多模型基準測試 =================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"🚀 開始執行時序強健性壓力測試 (Time-Warping Benchmark)")
    print(f"{'='*60}")
    
    # --- A. 載入訓練與驗證數據 ---
    X_train_raw, y_train_raw = load_sensor_data(os.path.join(BASE_PATH, 'training'))
    X_val_raw, y_val_raw     = load_sensor_data(os.path.join(BASE_PATH, 'validation'))
    
    # 載入所有測試集
    test_datasets = {}
    for factor, folder_name in test_folders.items():
        X_test, y_test = load_sensor_data(os.path.join(BASE_PATH, folder_name))
        test_datasets[factor] = (X_test, y_test)
    
    # --- B. 標籤轉換 ---
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_raw)
    y_val_encoded   = encoder.transform(y_val_raw)
    num_classes = len(encoder.classes_)
    
    y_train = to_categorical(y_train_encoded, num_classes)
    y_val   = to_categorical(y_val_encoded, num_classes)
    
    test_labels_cat = {}
    for factor, (X_test, y_test_raw) in test_datasets.items():
        y_test_encoded = encoder.transform(y_test_raw)
        test_labels_cat[factor] = to_categorical(y_test_encoded, num_classes)

    # --- C. 數據標準化 (Z-score) ---
    num_features = 5
    scaler = StandardScaler()
    
    # X_train_scaled = scaler.fit_transform(X_train_raw.reshape(-1, num_features)).reshape(-1, TARGET_LINES, num_features)
    # X_val_scaled   = scaler.transform(X_val_raw.reshape(-1, num_features)).reshape(-1, TARGET_LINES, num_features)
    
    test_features_scaled = {}
    for factor, (X_test_raw, _) in test_datasets.items():
        test_features_scaled[factor] = scaler.transform(X_test_raw.reshape(-1, num_features)).reshape(-1, TARGET_LINES, num_features)

    # --- D. 大亂鬥開始 ---
    for name in model_configs:
        print(f"\n🔹 [目前測試架構：{name}]")

        for run in range(num_runs):
            print(f"  ▶ 正在訓練第 {run + 1:02d}/{num_runs} 次...", end=" ", flush=True)
            
            # 建立模型
            if name == '1D-CNN':
                model = keras.Sequential([
                    keras.Input(shape=(TARGET_LINES, 5)),
                    keras.layers.Conv1D(filters=8, kernel_size=5, activation='relu'),
                    keras.layers.GlobalAveragePooling1D(),
                    keras.layers.Dense(5, activation='softmax')
                ])
            elif name == 'SimpleRNN-8':
                model = keras.Sequential([
                    keras.Input(shape=(TARGET_LINES, 5)),
                    keras.layers.SimpleRNN(8, return_sequences=False),
                    keras.layers.Dense(5, activation='softmax')
                ])
            elif name == 'LSTM-8':
                model = keras.Sequential([
                    keras.Input(shape=(TARGET_LINES, 5)),
                    keras.layers.LSTM(8, return_sequences=False),
                    keras.layers.Dense(5, activation='softmax')
                ])
            elif name == 'LTC-8':
                model = keras.Sequential([
                    keras.Input(shape=(TARGET_LINES, 5)),
                    keras.layers.RNN(LTCNeuron(units=8), return_sequences=False),
                    keras.layers.Dense(5, activation='softmax')
                ])

            custom_adam = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0)
            model.compile(optimizer=custom_adam, loss='categorical_crossentropy', metrics=['accuracy'], jit_compile=True)

            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=50, restore_best_weights=True, verbose=1
            )

            # 🔑 統一使用 500 epochs, batch_size=8
            history = model.fit(
                X_train_raw, y_train, 
                epochs=1000,
                batch_size=8, 
                validation_data=(X_val_raw, y_val),
                callbacks=[early_stopping],
                verbose=1 
            )
            
            # 🌟 訓練完畢，立刻連續考三張考卷！
            print("訓練完畢! 測試成績: ", end="")
            for factor in test_folders.keys():
                loss, accuracy = model.evaluate(test_features_scaled[factor], test_labels_cat[factor], verbose=0)
                all_test_accuracies[factor][name].append(accuracy * 100)
                print(f"[{factor}]:{accuracy*100:5.1f}% ", end="")
            print() # 換行
            
    # ================= 5. 將結果存檔 =================
    # 攤平字典存成 CSV
    flat_data = []
    for factor in test_folders.keys():
        for name in model_configs:
            for run, acc in enumerate(all_test_accuracies[factor][name]):
                flat_data.append({'Slowdown_Factor': factor, 'Model': name, 'Run': run+1, 'Accuracy': acc})
    
    df_results = pd.DataFrame(flat_data)
    df_results.to_csv("Benchmark_TimeWarping_Results.csv", index=False)
    print("\n✅ 所有結果已成功儲存至 Benchmark_TimeWarping_Results.csv")

    # ================= 繪製精美 Boxplot =================
    print("\n📊 正在繪製時間扭曲壓力測試 Boxplot...")
    
    # 修正 2 & 3：使用 Seaborn 繪製分組箱型圖 (Grouped Boxplot)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300) 
    
    custom_palette = {
        '1D-CNN': '#ff7f0e',
        'SimpleRNN-8': '#7f7f7f',
        'LSTM-8': '#d62728',
        'LTC-8': '#1f77b4'
    }
    
    # 確保 X 軸的顯示順序是 1.0x -> 2.0x -> 3.0x
    order_labels = ['1.0x', '2.0x', '3.0x']

    sns.boxplot(
        data=df_results, 
        x='Slowdown_Factor', 
        y='Accuracy', 
        hue='Model', 
        palette=custom_palette, 
        order=order_labels, 
        hue_order=['1D-CNN', 'SimpleRNN-8', 'LSTM-8', 'LTC-8'], 
        ax=ax,
        width=0.6,              
        fliersize=5,            
        linewidth=1.2           
    )

    # 設定標題與軸標籤
    ax.set_title(r"Zero-shot Time-Warping Robustness (10 Independent Runs)", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Time Slowdown Factor (1.0x = Original Speed)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Test Accuracy (%)", fontsize=14, fontweight='bold')
    
    # 動態調整 Y 軸，避免標籤被切掉
    min_acc = df_results['Accuracy'].min()
    ax.set_ylim([max(0, min_acc - 5), 105])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 調整圖例位置
    plt.legend(title='Neural Architecture', title_fontsize='12', fontsize='11', 
               loc='lower left', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig("Benchmark_Accuracy_Boxplot_TimeWarping.png", dpi=300, bbox_inches='tight')
    print("📊 完美！時間扭曲基準測試圖表已成功儲存為 Benchmark_Accuracy_Boxplot_TimeWarping.png")
    
    plt.show()