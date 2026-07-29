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

model_configs = ['1D-CNN', 'SimpleRNN-8', 'LSTM-8', 'LTC-4']

# 🌟 核心修改 1：設定平移級距 (負值為前移/提早發生，正值為後移/延遲發生)
shift_levels = ['Clean', -100, 100] 
num_runs = 10

# 建立儲存結果的字典
all_results = {model: {shift: [] for shift in shift_levels} for model in model_configs}

# ================= 2. 資料讀取與時間平移生成函式 =================
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

# 🌟 核心修改 2：時間平移函式 (Time Shift)
def add_time_shift(signal, shift_frames):
    """
    對時間序列數據進行平移，並使用邊緣填充(Edge Padding)維持物理連續性
    signal shape: (樣本數, 時間步 400, 通道數 5)
    """
    if shift_frames == 'Clean' or shift_frames == 0:
        return signal
        
    shifted_signal = np.empty_like(signal)
    
    for i in range(signal.shape[0]):
        for c in range(signal.shape[2]):
            trace = signal[i, :, c]
            if shift_frames > 0:
                # 正值：訊號後移 (延遲發生)。前面空缺的部分，用原本的第 0 個數值填補
                shifted_signal[i, :, c] = np.concatenate([np.full(shift_frames, trace[0]), trace[:-shift_frames]])
            else:
                # 負值：訊號前移 (提早發生)。後面空缺的部分，用原本的最後 1 個數值填補
                s = abs(shift_frames)
                shifted_signal[i, :, c] = np.concatenate([trace[s:], np.full(s, trace[-1])])
                
    return shifted_signal

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

# ================= 4. 主程式：時間平移強健性測試 =================
if __name__ == "__main__":
    print(f"\n{'='*75}")
    print(f"⏱️ 開始執行：動態軌跡模型 時間平移抗性測試 (Time Shift Robustness)")
    print(f"{'='*75}")
    
    X_train_raw, y_train_raw = load_sensor_data(os.path.join(BASE_PATH, 'training'))
    X_val_raw, y_val_raw = load_sensor_data(os.path.join(BASE_PATH, 'validation'))
    X_test_raw, y_test_raw = load_sensor_data(os.path.join(BASE_PATH, 'test'))
    
    encoder = LabelEncoder()
    encoder.fit(y_train_raw)
    num_classes = len(encoder.classes_)
    num_features = 3 

    print(f"通道有: {num_features}\n")
    print(f"類別有: {num_classes}\n")

    y_train = to_categorical(encoder.transform(y_train_raw), num_classes)
    y_val = to_categorical(encoder.transform(y_val_raw), num_classes)
    y_test = to_categorical(encoder.transform(y_test_raw), num_classes)

    for name in model_configs:
        print(f"\n🔹 [目前測試架構：{name}]")
        
        for run in range(num_runs):
            print(f"   ▶ Run {run + 1:02d}/{num_runs} (Training)...", end=" ", flush=True)
            
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

            history = model.fit(
                X_train_raw, y_train, 
                epochs=2000,
                batch_size=8, 
                validation_data=(X_val_raw, y_val),
                callbacks=[early_stopping],
                verbose=0 
            )
            print("訓練完成！進入 Time Shift 壓力測試:")

            # 🌟 測試不同平移級距
            for shift in shift_levels:
                X_test_shifted = add_time_shift(X_test_raw, shift)
                
                loss, accuracy = model.evaluate(X_test_shifted, y_test, verbose=0)
                all_results[name][shift].append(accuracy * 100)
                
                shift_label = "Clean" if shift == 'Clean' else f"{shift:+d} F"
                print(f"      - Shift: {shift_label:>8} -> Acc: {accuracy*100:>5.2f}%")

    print("\n✅ 所有實驗完畢！開始整理數據與繪製平移測試分組箱型圖...")

    # ================= 5. 數據轉換與分組箱型圖繪製 =================
    flat_results = []
    for name in model_configs:
        for shift in shift_levels:
            for i, acc in enumerate(all_results[name][shift]):
                # 將標籤格式化，例如 '-25 Frames' 或 '+50 Frames'
                if shift == 'Clean':
                    label = "Clean"
                elif shift == -100:
                    label = "-1s"
                elif shift == 100:
                    label = "+1s"
                else:
                    label = f"{shift:+d}s"

                flat_results.append({
                    'Model': name, 
                    'Shift Level': label, 
                    'Run': i+1, 
                    'Accuracy (%)': acc
                })
    df_results = pd.DataFrame(flat_results)
    
    df_results.to_csv("TimeShift_Robustness_Raw_Data.csv", index=False)
    print("💾 原始測試數據已儲存為 TimeShift_Robustness_Raw_Data.csv")

    sns.set_theme(style="whitegrid") 
    fig, ax = plt.subplots(figsize=(15, 7), dpi=300)
    
    model_palette = {
        '1D-CNN': '#ff7f0e', 
        'SimpleRNN-8': '#7f7f7f',
        'LSTM-8': '#d62728',
        'LTC-4': '#1f77b4'
    }

    # 確保 X 軸按照我們設定的順序排列
    x_order = ['Clean', '-1s', '+1s']
    hue_order = ['1D-CNN', 'SimpleRNN-8', 'LSTM-8', 'LTC-4']

    sns.boxplot(
        data=df_results, 
        x='Shift Level', 
        y='Accuracy (%)', 
        hue='Model', 
        order=x_order,
        hue_order=hue_order,
        ax=ax, 
        palette=model_palette,
        linewidth=1.5,
        width=0.7,
        fliersize=4
    )

    # 🔑 加入平均值標籤
    offsets = [-0.2625, -0.0875, 0.0875, 0.2625]
    global_min = df_results['Accuracy (%)'].min()
    # text_y_pos = max(0, global_min - 3.0) 

    for x_idx, shift_label in enumerate(x_order):
        for hue_idx, model in enumerate(hue_order):
            subset = df_results[(df_results['Shift Level'] == shift_label) & (df_results['Model'] == model)]
            if not subset.empty:
                mean_val = subset['Accuracy (%)'].mean()
                min_val = subset['Accuracy (%)'].min()
                x_pos = x_idx + offsets[hue_idx]  
                
                ax.text(x_pos, min_val, f'{mean_val:.1f}', 
                        ha='center', va='top', fontsize=9, fontweight='bold', color='black',
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='#CCCCCC', pad=1.5),
                        zorder=10)

    # 設定標題與軸標籤
    ax.set_title("OOD Robustness: Model Accuracy Under Temporal Shift", fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("Time Shift (Frames)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Test Accuracy (%)", fontsize=14, fontweight='bold')
    
    ax.set_ylim([max(0, global_min - 12), 100])
    
    plt.legend(title='Model Architecture', title_fontsize='12', fontsize='11', loc='lower left')

    plt.tight_layout()
    plt.savefig("TimeShift_Robustness_Grouped_Boxplot.png", dpi=300, bbox_inches='tight')
    print("📊 完美！時間平移測試圖表已成功儲存為 TimeShift_Robustness_Grouped_Boxplot.png")
    
    plt.show()