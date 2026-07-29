import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical

# ================= 1. 全局設定 =================
TARGET_LINES = 400
# 🚨 請確認資料夾路徑是否正確
BASE_PATH = r'C:\Users\HAO\Desktop\YTY_from_macbook\dataset_xx2020\dataset_602020_zeroed' 

# 💡 乾淨俐落，只留主角
model_configs = ['LTC-4']
num_runs = 1

all_test_accuracies = {name: [] for name in model_configs}

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
        
        # static_bias = tf.reduce_sum(self.w * tf.math.sigmoid(self.mu), axis=0)
        # driving_adjusted = driving - static_bias

        dx = -damping * x + driving
        x_new = x + (delta_t * dx)
        
        return x_new, [x_new]

# ================= 4. LTC 可解釋性與物理動態繪圖函式 =================
def plot_ltc_interpretability(trained_model, X_test_scaled):
    print("\n   🔍 [XAI 觸發] 正在擷取 LTC-4 內部動態時間常數與神經元狀態...")
    
    inputs = keras.Input(shape=(TARGET_LINES, 5))
    transparent_rnn = keras.layers.RNN(LTCNeuron(units=4), return_sequences=True)
    hidden_states = transparent_rnn(inputs)
    
    extractor = keras.Model(inputs=inputs, outputs=hidden_states)
    extractor.layers[1].set_weights(trained_model.layers[0].get_weights())
    
    # 挑選測試集的第一個樣本
    sample_index = 0 
    sample_gesture = X_test_scaled[sample_index:sample_index+1]
    
    extracted_states = extractor.predict(sample_gesture, verbose=0)
    states_trajectory = extracted_states[0] 
    sensor_trajectory = sample_gesture[0]   
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, dpi=300)

    # 上半部：真實感測器波形
    fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    colors_sensor = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i in range(5):
        ax1.plot(sensor_trajectory[:, i], label=f'{fingers[i]} Sensor', color=colors_sensor[i], linewidth=2, alpha=0.8)

    ax1.set_title("Input Sub-System: Raw Sensor Trajectories", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Normalized Voltage", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=10, ncol=5)

    # 下半部：LTC 神經元內部狀態
    colors_neuron = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    for i in range(4):
        ax2.plot(states_trajectory[:, i], label=f'LTC Neuron {i+1}', color=colors_neuron[i], linewidth=2.5)

    ax2.set_title("ODE Hidden Sub-System: LTC Internal State Dynamics ($x_t$)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Time Steps (400 samples)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Internal State Activation", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=10, ncol=4)

    # ax1.set_xlim(120, 400) # 把 X 軸限制在 120 步到 400 步
    # ax2.set_xlim(120, 400)

    plt.tight_layout()
    output_filename = "LTC4_Interpretability_Dynamics.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig) 
    print(f"   ✅ XAI 物理動態展示圖表已背景儲存為：{output_filename}\n")

# ================= 5. 主程式：LTC-4 專屬訓練 =================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"🚀 開始執行 LTC-4 專屬訓練與物理動態擷取")
    print(f"{'='*60}")
    
    X_train_raw, y_train_raw = load_sensor_data(os.path.join(BASE_PATH, 'training'))
    X_val_raw, y_val_raw     = load_sensor_data(os.path.join(BASE_PATH, 'validation'))
    X_test_raw, y_test_raw   = load_sensor_data(os.path.join(BASE_PATH, 'test'))
    
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_raw)
    y_val_encoded   = encoder.transform(y_val_raw)
    y_test_encoded  = encoder.transform(y_test_raw)

    num_classes = len(encoder.classes_)
    y_train = to_categorical(y_train_encoded, num_classes)
    y_val   = to_categorical(y_val_encoded, num_classes)
    y_test  = to_categorical(y_test_encoded, num_classes)

    # ✅ 改成這樣就好：直接繼承完美的外部預處理資料！
    X_train_scaled = X_train_raw
    X_val_scaled   = X_val_raw
    X_test_scaled  = X_test_raw

    for name in model_configs:
        print(f"\n🔹 [目前測試架構：{name}]")

        for run in range(num_runs):
            print(f"  ▶ 開始第 {run + 1:02d}/{num_runs} 次訓練...", end=" ", flush=True)
            
            # 只保留 LTC-4
            model = keras.Sequential([
                keras.Input(shape=(TARGET_LINES, 5)),
                keras.layers.RNN(LTCNeuron(units=8), return_sequences=False),
                keras.layers.Dense(3, activation='softmax')
            ])

            custom_adam = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0)
            model.compile(optimizer=custom_adam, loss='categorical_crossentropy', metrics=['accuracy'], jit_compile=True)

            if(run == 0):
                model.summary()

            model.fit(
                X_train_scaled, y_train, 
                epochs=500,
                batch_size=8, 
                validation_data=(X_val_scaled, y_val),
                verbose=1
            )

            loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
            all_test_accuracies[name].append(accuracy * 100)
            print(f"完成！Test Acc: {accuracy*100:>6.2f}%")

            # 🌟 在第一回合訓練結束後，自動觸發 XAI 可視化腳本
            if run == 0:
                plot_ltc_interpretability(model, X_test_scaled)
            
        print(f"📊 {name} 10次平均準確率: {np.mean(all_test_accuracies[name]):.2f}%\n")

    # ================= 專屬存檔與簡單視覺化 =================
    df_results = pd.DataFrame(all_test_accuracies)
    df_results.to_csv("LTC4_Only_Results.csv", index=False)
    print("\n✅ LTC-4 專屬數據已安全儲存至 LTC4_Only_Results.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    bplot = ax.boxplot([all_test_accuracies['LTC-4']], patch_artist=True, labels=['LTC-4'], 
                       boxprops=dict(facecolor='#1f77b4', color='black', linewidth=1.5, alpha=0.6),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5))

    avg_acc = np.mean(all_test_accuracies['LTC-4'])
    y_min, y_max = ax.get_ylim()
    ax.text(1, y_min + (y_max - y_min) * 0.05, f'Mean: {avg_acc:.2f}%', 
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.9))

    ax.set_title("LTC-4 Standalone Performance (10 Runs)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("LTC4_Only_Boxplot.png", dpi=300, bbox_inches='tight')
    print("📊 專屬 Boxplot 已儲存為 LTC4_Only_Boxplot.png")
    
    plt.show()