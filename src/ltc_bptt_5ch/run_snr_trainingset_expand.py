import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import tempfile
import seaborn as sns 

# 讓終端機保持乾淨
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ================= 1. 全局設定 =================
TARGET_LINES = 400
BASE_PATH = r'C:\Users\HAO\Desktop\YTY_from_macbook\dataset_xx2020\dataset_602020_zscore' 

# model_configs = ['1D-CNN', 'SimpleRNN-8', 'LSTM-8', 'LTC-4', 'LTC-8']
model_configs = ['LTC-4', 'LTC-8']


# 擴增用的雜訊級距 (擴展 5 倍資料)
snr_levels = ['Clean', 30, 20, 10, 0] 
num_runs = 10 

# 儲存結果的字典：這次只要記下每個模型 10 次的 Test Accuracy 即可
all_results = {model: [] for model in model_configs}

# ================= 2. 資料處理與雜訊擴增函式 =================
def load_sensor_data(folder_path):
    all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    signal_list, label_list = [], []
    for file in all_files:
        filename = os.path.basename(file)
        label = filename.rsplit('_', 1)[0]
        df = pd.read_csv(file)
        features = df[['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']].values
        if features.shape == (TARGET_LINES, 5):
            signal_list.append(features)
            label_list.append(label)
    return np.array(signal_list), np.array(label_list)

def add_awgn_noise(signal, snr_db):
    if snr_db == 'Clean':
        return signal
    sig_power = np.mean(signal ** 2)
    sig_power_db = 10 * np.log10(sig_power + 1e-10)
    noise_power_db = sig_power_db - snr_db
    noise_power = 10 ** (noise_power_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def augment_dataset(X, y, snr_list):
    """將傳入的資料集擴增 (複製並加入不同雜訊)"""
    X_aug, y_aug = [], []
    for snr in snr_list:
        X_noisy = add_awgn_noise(X, snr)
        X_aug.append(X_noisy)
        y_aug.append(y)
        
    X_aug = np.concatenate(X_aug, axis=0)
    y_aug = np.concatenate(y_aug, axis=0)
    
    indices = np.arange(len(X_aug))
    np.random.shuffle(indices)
    return X_aug[indices], y_aug[indices]

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

# ================= 4. 主程式：擴增資料基準測試 =================
if __name__ == "__main__":
    print(f"\n{'='*65}")
    print(f"🏆 開始執行：擴增訓練集 (5x Augmented) 之模型架構基準測試")
    print(f"{'='*65}")
    
    X_train_raw, y_train_raw = load_sensor_data(os.path.join(BASE_PATH, 'training'))
    X_val_raw, y_val_raw = load_sensor_data(os.path.join(BASE_PATH, 'validation'))
    X_test_raw, y_test_raw = load_sensor_data(os.path.join(BASE_PATH, 'test'))
    
    encoder = LabelEncoder()
    encoder.fit(y_train_raw)
    num_classes = len(encoder.classes_)
    
    # 資料已做過 Z-score，直接套用
    X_train_scaled = X_train_raw
    X_val_scaled = X_val_raw
    X_test_scaled = X_test_raw
    
    y_train_base = to_categorical(encoder.transform(y_train_raw), num_classes)
    y_val_base = to_categorical(encoder.transform(y_val_raw), num_classes)
    y_test = to_categorical(encoder.transform(y_test_raw), num_classes)

    print("🧬 正在執行資料擴增 (Data Augmentation)...")
    X_train_aug, y_train_aug = augment_dataset(X_train_scaled, y_train_base, snr_levels)
    X_val_aug, y_val_aug = augment_dataset(X_val_scaled, y_val_base, snr_levels)
    print(f"   - 訓練集大小已擴增至: {len(X_train_aug)} 筆")
    print(f"   - 驗證集大小已擴增至: {len(X_val_aug)} 筆\n")

    train_batch_size = 8

    for name in model_configs:
        print(f"🔹 [目前測試架構：{name}]")
        
        for run in range(num_runs):
            print(f"   ▶ Run {run + 1:02d}/{num_runs}...", end=" ", flush=True)
            
            # 建立模型
            if name == '1D-CNN':
                model = keras.Sequential([
                    keras.Input(shape=(TARGET_LINES, 5)),
                    keras.layers.Conv1D(filters=8, kernel_size=5, activation='relu'),
                    keras.layers.GlobalAveragePooling1D(),
                    keras.layers.Dense(num_classes, activation='softmax')
                ])
            elif name == 'SimpleRNN-8':
                model = keras.Sequential([
                    keras.Input(shape=(TARGET_LINES, 5)),
                    keras.layers.SimpleRNN(8, return_sequences=False),
                    keras.layers.Dense(num_classes, activation='softmax')
                ])
            elif name == 'LSTM-8':
                model = keras.Sequential([
                    keras.Input(shape=(TARGET_LINES, 5)),
                    keras.layers.LSTM(8, return_sequences=False),
                    keras.layers.Dense(num_classes, activation='softmax')
                ])
            elif name == 'LTC-4':
                model = keras.Sequential([
                    keras.Input(shape=(TARGET_LINES, 5)),
                    keras.layers.RNN(LTCNeuron(units=4), return_sequences=False),
                    keras.layers.Dense(num_classes, activation='softmax')
                ])
            elif name == 'LTC-8':
                model = keras.Sequential([
                    keras.Input(shape=(TARGET_LINES, 5)),
                    keras.layers.RNN(LTCNeuron(units=8), return_sequences=False),
                    keras.layers.Dense(num_classes, activation='softmax')
                ])

            model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0), 
                          loss='categorical_crossentropy', metrics=['accuracy'], jit_compile=True)

            # 在巨大的擴增資料集上訓練
            model.fit(X_train_aug, y_train_aug, epochs=500, batch_size=train_batch_size, 
                      validation_data=(X_val_aug, y_val_aug), verbose=0)

            # 🌟 測試時，直接在原本乾淨的 Test Set 上看最終實力
            loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
            all_results[name].append(accuracy * 100)
            
            print(f"Test Acc: {accuracy*100:>5.2f}%")

    print("\n✅ 所有實驗完畢！開始繪製 Benchmark 箱型圖...")

    # ================= 5. 繪製 Benchmark 箱型圖 (Boxplot) =================
    flat_results = []
    for name in model_configs:
        for i, acc in enumerate(all_results[name]):
            flat_results.append({'Model': name, 'Run': i+1, 'Accuracy': acc})
    df_results = pd.DataFrame(flat_results)
    
    df_results.to_csv("Benchmark_Augmented_Raw_Data.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    custom_palette = {'1D-CNN': '#c4b5fd', 'SimpleRNN-8': '#d4a5a5', 'LSTM-8': '#f9a8d4', 'LTC-4': '#93c5fd', 'LTC-8': '#fdba74'}

    # 畫箱型圖，這次 X 軸直接是模型
    sns.boxplot(
        data=df_results, x='Model', y='Accuracy', 
        palette=custom_palette, order=model_configs,
        ax=ax, width=0.5, fliersize=5, linewidth=1.5 
    )

    # 🌟 在每個箱子的下方加入 Mean (平均值) 的文字標籤 (對齊你的 PDF)
    means = df_results.groupby('Model')['Accuracy'].mean().reindex(model_configs)
    min_acc = df_results['Accuracy'].min()
    y_text_pos = max(0, min_acc - 3) # 把文字放在最低點再往下一點點

    for i, model_name in enumerate(model_configs):
        mean_val = means[model_name]
        ax.text(i, y_text_pos, f'Mean: {mean_val:.2f}%', 
                ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='#cccccc', boxstyle='round,pad=0.3'))

    ax.set_title("Model Architecture Performance Comparison (Trained on 5x Augmented Data)", 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Neural Architecture", fontsize=14, fontweight='bold')
    ax.set_ylabel("Test Accuracy (%)", fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', labelsize=12)
    
    # 調整 Y 軸範圍讓下面的文字標籤不會被切掉
    ax.set_ylim(max(0, y_text_pos - 3), 105)

    plt.tight_layout()
    plt.savefig("Benchmark_Augmented_Boxplot.png", dpi=300, bbox_inches='tight')
    print("📊 完美！Benchmark 箱型圖已儲存為 Benchmark_Augmented_Boxplot.png")
    plt.show()