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
import seaborn as sns 

# 讓終端機保持乾淨
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ================= 1. 全局設定 =================
TARGET_LINES = 400
BASE_PATH = r'C:\Users\HAO\Desktop\YTY_from_macbook\dataset_xx2020\dataset_602020_zscore' 

model_configs = ['1D-CNN', 'SimpleRNN-8', 'LSTM-8', 'LTC-4', 'LTC-8']

# 🌟 新增：設定不同的掉包率 (Drop Rates)
# 0.0 代表無掉包，0.5 代表高達 50% 的時間點沒有訊號
drop_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] 
num_runs = 10 

all_results = {model: {rate: [] for rate in drop_rates} for model in model_configs}

# ================= 2. 資料讀取與掉包模擬函式 =================
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

# 🌟 關鍵新增：模擬藍牙掉封包 (Packet Loss)
def simulate_packet_loss(signal, drop_rate):
    """
    根據指定的 drop_rate 隨機將某些時間步的資料歸零。
    為了符合真實 IoT 裝置行為，同一個時間步的 5 個通道會同時掉包。
    """
    if drop_rate == 0.0:
        return signal
    
    # 產生一個長度為 400 的隨機數陣列
    random_probs = np.random.rand(signal.shape[0], 1)
    
    # 如果隨機數大於 drop_rate，就保留 (True, 1)；否則就掉包 (False, 0)
    # 形狀為 (400, 1)，這樣乘上去時，Numpy 會自動廣播到 5 個手指通道
    mask = (random_probs > drop_rate).astype(float)
    
    # 將掉包的時間點數值強制設為 0 (回歸 Baseline)
    dropped_signal = signal * mask
    
    return dropped_signal

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

# ================= 4. 主程式：掉封包強健性測試 =================
if __name__ == "__main__":
    print(f"\n{'='*65}")
    print(f"📡 開始執行：IoT 藍牙掉封包 (Packet Loss) 極限測試")
    print(f"{'='*65}")
    
    X_train_raw, y_train_raw = load_sensor_data(os.path.join(BASE_PATH, 'training'))
    X_val_raw, y_val_raw = load_sensor_data(os.path.join(BASE_PATH, 'validation'))
    X_test_raw, y_test_raw = load_sensor_data(os.path.join(BASE_PATH, 'test'))
    
    encoder = LabelEncoder()
    encoder.fit(y_train_raw)
    num_classes = len(encoder.classes_)
    
    num_features = 5
    X_train_scaled = X_train_raw
    X_val_scaled = X_val_raw
    X_test_scaled = X_test_raw
    
    y_train = to_categorical(encoder.transform(y_train_raw), num_classes)
    y_val = to_categorical(encoder.transform(y_val_raw), num_classes)
    y_test = to_categorical(encoder.transform(y_test_raw), num_classes)

    train_batch_size = 32 

    # 外層迴圈 1：切換模型
    for name in model_configs:
        print(f"\n🔹 [目前測試架構：{name}]")
        
        for run in range(num_runs):
            print(f"   ▶ Run {run + 1:02d}/{num_runs} (Training on Perfect Connection)...", end=" ", flush=True)
            
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

            # 在「沒有掉包」的完美資料上訓練
            model.fit(X_train_scaled, y_train, epochs=500, batch_size=train_batch_size, verbose=0)
            print("訓練完成！進入斷訊壓力測試:")

            # 🌟 內層迴圈：在不同的掉包率下考驗模型
            for rate in drop_rates:
                # 幫 Test Set 的每一筆資料隨機挖空掉包
                # 注意：X_test_scaled 的形狀是 (樣本數, 400, 5)
                X_test_dropped = np.array([simulate_packet_loss(sample, rate) for sample in X_test_scaled])
                
                loss, accuracy = model.evaluate(X_test_dropped, y_test, verbose=0)
                all_results[name][rate].append(accuracy * 100)
                
                print(f"      - Drop Rate: {rate*100:>3.0f}% -> Acc: {accuracy*100:>5.2f}%")

    print("\n✅ 所有實驗完畢！開始整理數據與繪製掉封包箱型圖...")

    # ================= 5. 數據轉換與箱型圖 (Boxplot) 繪製 =================
    flat_results = []
    for name in model_configs:
        for rate in drop_rates:
            for i, acc in enumerate(all_results[name][rate]):
                flat_results.append({
                    'Model': name, 
                    'Packet Loss Rate (%)': int(rate * 100), 
                    'Run': i+1, 
                    'Accuracy': acc
                })
    df_results = pd.DataFrame(flat_results)
    
    df_results.to_csv("Packet_Loss_Robustness_Raw_Data.csv", index=False)
    print("💾 原始測試數據已儲存為 Packet_Loss_Robustness_Raw_Data.csv")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    custom_palette = {'1D-CNN': '#ff7f0e', 'SimpleRNN-8': '#7f7f7f', 'LSTM-8': '#d62728', 'LTC-4': '#2ca02c', 'LTC-8': '#1f77b4'}

    order_labels = [int(rate * 100) for rate in drop_rates]

    sns.boxplot(
        data=df_results, 
        x='Packet Loss Rate (%)', 
        y='Accuracy', 
        hue='Model', 
        palette=custom_palette, 
        order=order_labels, 
        hue_order=['1D-CNN', 'SimpleRNN-8', 'LSTM-8', 'LTC-4', 'LTC-8'], 
        ax=ax,
        width=0.6,              
        fliersize=5,            
        linewidth=1.2           
    )

    ax.set_title("Model Robustness against Bluetooth Packet Loss (10 Runs)", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Packet Loss Rate (%)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Test Accuracy (%)", fontsize=14, fontweight='bold')
    
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', labelsize=12)
    
    min_acc = df_results['Accuracy'].min()
    ax.set_ylim(max(0, min_acc - 5), 105)

    plt.legend(title='Neural Architecture', title_fontsize='12', fontsize='11', loc='lower left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig("Packet_Loss_Robustness_Boxplot.png", dpi=300, bbox_inches='tight')
    print("📊 完美！掉封包箱型圖已儲存為 Packet_Loss_Robustness_Boxplot.png")
    
    plt.show()