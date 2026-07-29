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

# 讓終端機保持乾淨，消除不必要的警告
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ================= 1. 全局設定 =================
TARGET_LINES = 400
BASE_PATH = r'C:\Users\HAO\Desktop\YTY_from_macbook\dataset_xx2020_new_new_new\dataset_602020_zscore' 

# 🌟 統一使用 LTC-8 作為主力戰將
model_configs = ['1D-CNN', 'SimpleRNN-8', 'LSTM-8', 'LTC-4']

# 🌟 換成信噪比 (SNR) 級距。'Clean' 代表無雜訊，數值越小雜訊越強 (0dB 為極限)
snr_levels = ['Clean', 15, 5] 
num_runs = 10

# 建立儲存結果的字典
all_results = {model: {snr: [] for snr in snr_levels} for model in model_configs}

# ================= 2. 資料讀取與 SNR 雜訊生成函式 =================
def load_sensor_data(folder_path):
    all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    signal_list, label_list = [], []
    for file in all_files:
        filename = os.path.basename(file)
        label = filename.rsplit('_', 1)[0]
        df = pd.read_csv(file)
        # 若為三通道消融實驗，請改為 ['Thumb', 'Middle', 'Pinky']
        features = df[['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']].values
        if features.shape == (TARGET_LINES, 5):
            signal_list.append(features)
            label_list.append(label)
    return np.array(signal_list), np.array(label_list)

# 🌟 核心修改：產生指定 SNR (dB) 的高斯白雜訊 (AWGN)
def add_snr_noise(signal, snr_db):
    """
    對時間序列數據加入指定 SNR (dB) 的高斯白雜訊
    signal shape: (樣本數, 時間步 400, 通道數 5)
    """
    if snr_db == 'Clean':
        return signal
        
    noisy_signal = np.zeros_like(signal)
    for i in range(signal.shape[0]):
        # 1. 計算單筆樣本的訊號功率 (Signal Power)
        signal_power = np.mean(signal[i] ** 2)
        signal_power_db = 10 * np.log10(signal_power + 1e-8) 
        
        # 2. 根據目標 SNR 計算所需的雜訊功率
        noise_power_db = signal_power_db - snr_db
        noise_power = 10 ** (noise_power_db / 10)
        
        # 3. 產生高斯雜訊並疊加
        noise = np.random.normal(0, np.sqrt(noise_power), signal[i].shape)
        noisy_signal[i] = signal[i] + noise
        
    return noisy_signal

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

# ================= 4. 主程式：OOD 雜訊強健性測試 =================
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"🌪️ 開始執行：動態軌跡模型 OOD 抗噪能力測試 (SNR Robustness Test)")
    print(f"{'='*70}")
    
    X_train_raw, y_train_raw = load_sensor_data(os.path.join(BASE_PATH, 'training'))
    X_val_raw, y_val_raw = load_sensor_data(os.path.join(BASE_PATH, 'validation'))
    X_test_raw, y_test_raw = load_sensor_data(os.path.join(BASE_PATH, 'test'))
    
    encoder = LabelEncoder()
    encoder.fit(y_train_raw)
    num_classes = len(encoder.classes_)
    
    # 預設 5 通道。若為消融實驗請改為 3
    num_features = 5 

    print(f"類別有: {num_classes}\n")

    y_train = to_categorical(encoder.transform(y_train_raw), num_classes)
    y_val = to_categorical(encoder.transform(y_val_raw), num_classes)
    y_test = to_categorical(encoder.transform(y_test_raw), num_classes)

    # 外層迴圈 1：切換模型
    for name in model_configs:
        print(f"\n🔹 [目前測試架構：{name}]")
        
        # 外層迴圈 2：進行 N 次獨立隨機訓練
        for run in range(num_runs):
            print(f"   ▶ Run {run + 1:02d}/{num_runs} (Training)...", end=" ", flush=True)
            
            # 建立模型
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

            custom_adam = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0)
            model.compile(optimizer=custom_adam, loss='categorical_crossentropy', metrics=['accuracy'], jit_compile=True)

            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=200, restore_best_weights=True, verbose=0
            )

            # 訓練模型 (於乾淨數據上)
            history = model.fit(
                X_train_raw, y_train, 
                epochs=2000,
                batch_size=8, 
                validation_data=(X_val_raw, y_val),
                callbacks=[early_stopping],
                verbose=0 
            )
            print("訓練完成！進入 SNR 壓力測試:")

            # 🌟 內層迴圈：拿同一個訓練好的模型，考不同 SNR 級距的雜訊
            for snr in snr_levels:
                # 動態加入 SNR 雜訊至測試集
                X_test_noisy = add_snr_noise(X_test_raw, snr)
                
                loss, accuracy = model.evaluate(X_test_noisy, y_test, verbose=0)
                all_results[name][snr].append(accuracy * 100)
                
                snr_label = "Clean" if snr == 'Clean' else f"{snr} dB"
                print(f"      - Noise: {snr_label:>8} -> Acc: {accuracy*100:>5.2f}%")

    print("\n✅ 所有實驗完畢！開始整理數據與繪製抗噪分組箱型圖...")

    # ================= 5. 數據轉換與分組箱型圖 (Grouped Boxplot) 繪製 =================
    flat_results = []
    for name in model_configs:
        for snr in snr_levels:
            for i, acc in enumerate(all_results[name][snr]):
                label = "Clean" if snr == 'Clean' else f"{snr} dB"
                flat_results.append({
                    'Model': name, 
                    'SNR Level': label, 
                    'Run': i+1, 
                    'Accuracy (%)': acc
                })
    df_results = pd.DataFrame(flat_results)
    
    # 存 CSV
    df_results.to_csv("SNR_Robustness_Raw_Data.csv", index=False)
    print("💾 原始測試數據已儲存為 SNR_Robustness_Raw_Data.csv")

    # 🌟 完美重寫的 Seaborn 分組箱型圖
    sns.set_theme(style="whitegrid") # 使用乾淨的學術網格背景
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    
    # 自定義各模型的專屬顏色
    model_palette = {
        '1D-CNN': '#ff7f0e',
        'SimpleRNN-8': '#7f7f7f',
        'LSTM-8': '#d62728',
        'LTC-4': '#1f77b4'
    }

    # 繪製分組箱型圖
    sns.boxplot(
        data=df_results, 
        x='SNR Level', 
        y='Accuracy (%)', 
        hue='Model', 
        ax=ax, 
        palette=model_palette,
        linewidth=1.5,
        width=0.7,
        fliersize=4
    )

    x_order = ['Clean', '45 dB', '30 dB', '15 dB']
    hue_order = ['1D-CNN', 'SimpleRNN-8', 'LSTM-8', 'LTC-4']
    
    # 🔑 針對 width=0.7 且有 4 個模型的 Seaborn 完美平移比例
    # 計算方式：0.7 / 4 = 0.175，由中心向兩側推算
    offsets = [-0.2625, -0.0875, 0.0875, 0.2625]
    
    # 找出整張圖的全局最低點，讓所有文字對齊在同一水平線上
    global_min = df_results['Accuracy (%)'].min()
    text_y_pos = max(0, global_min - 3.0) # 往下推一點，留出呼吸空間

    for x_idx, snr_label in enumerate(x_order):
        for hue_idx, model in enumerate(hue_order):
            # 抓取該 SNR 級距與特定模型的資料子集
            subset = df_results[(df_results['SNR Level'] == snr_label) & (df_results['Model'] == model)]
            if not subset.empty:
                mean_val = subset['Accuracy (%)'].mean()
                x_pos = x_idx + offsets[hue_idx]   # 套用精準的 X 軸偏移
                
                # 統一 Y 座標，畫出整齊的標籤
                ax.text(x_pos, text_y_pos, f'{mean_val:.1f}', 
                        ha='center', va='top', fontsize=9, fontweight='bold', color='black',
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=1),
                        zorder=10)

    # 設定標題與軸標籤 (字體加大增加學術感)
    ax.set_title("OOD Robustness: Model Accuracy Across Different SNR Levels", fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Test Accuracy (%)", fontsize=14, fontweight='bold')
    
    # 動態調整 Y 軸，讓底部留有一點空間
    y_min = df_results['Accuracy (%)'].min()
    ax.set_ylim([max(0, y_min - 10), 105])
    
    # 調整圖例位置
    plt.legend(title='Model Architecture', title_fontsize='12', fontsize='11', loc='lower left')

    plt.tight_layout()
    plt.savefig("SNR_Robustness_Grouped_Boxplot.png", dpi=300, bbox_inches='tight')
    print("📊 完美！SNR 抗噪測試圖表已成功儲存為 SNR_Robustness_Grouped_Boxplot.png")
    
    plt.show()