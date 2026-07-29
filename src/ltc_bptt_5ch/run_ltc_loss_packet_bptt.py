import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from sklearn.preprocessing import LabelEncoder
import keras
import tempfile

# 讓終端機保持乾淨，避免 TF 警告訊息洗版
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

# ================= 全局設定 =================
TARGET_LINES = 400
BASE_PATH = r'C:\Users\HAO\Desktop\YTY_from_macbook\dataset_xx2020\dataset_602020_zscore' 

# ================= 1. 資料讀取 =================
def load_sensor_data(folder_path):
    all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    signal_list, label_list = [] , []
    for file in all_files:
        filename = os.path.basename(file)
        label = filename.rsplit('_', 1)[0]
        df = pd.read_csv(file)
        features = df[['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']].values
        if features.shape == (TARGET_LINES, 5):
            signal_list.append(features)
            label_list.append(label)
    return np.array(signal_list), np.array(label_list)

def encode_label(label_train, label_val, label_test):
    encoder = LabelEncoder()
    label_train_encoded = encoder.fit_transform(label_train)
    label_val_encoded   = encoder.transform(label_val)
    label_test_encoded  = encoder.transform(label_test)
    num_classes = len(encoder.classes_)
    y_train = np.eye(num_classes)[label_train_encoded]
    y_val   = np.eye(num_classes)[label_val_encoded]
    y_test  = np.eye(num_classes)[label_test_encoded]
    return y_train, y_val, y_test

print("🚀 載入資料中...")
(X_train_raw, y_train_raw) = load_sensor_data(os.path.join(BASE_PATH, 'training'))
(X_val_raw, y_val_raw)     = load_sensor_data(os.path.join(BASE_PATH, 'validation'))
(X_test_raw, y_test_raw)   = load_sensor_data(os.path.join(BASE_PATH, 'test'))

Y_train, Y_val, Y_test = encode_label(y_train_raw, y_val_raw, y_test_raw)
X_train, X_val, X_test = X_train_raw.astype(np.float32), X_val_raw.astype(np.float32), X_test_raw.astype(np.float32)

def get_few_shot_subset(X, Y, raw_labels, samples_per_class):
    unique_classes = np.unique(raw_labels)
    selected_indices = []
    for cls in unique_classes:
        cls_indices = np.where(raw_labels == cls)[0]
        selected_indices.extend(cls_indices[:samples_per_class])
    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices) 
    return X[selected_indices], Y[selected_indices]

# ================= 2. 模擬掉封包 (Zero-Order Hold) =================
def apply_packet_loss(X_batch, drop_rate):
    """
    模擬遠端遙控掉封包：若該時間步掉包，感測器數值卡在上一幀 (Zero-Order Hold)
    """
    if drop_rate == 0.0:
        return X_batch.copy()
        
    X_dropped = X_batch.copy()
    batch_size, time_steps, features = X_dropped.shape
    
    for i in range(batch_size):
        for t in range(1, time_steps):
            if np.random.rand() < drop_rate:
                X_dropped[i, t, :] = X_dropped[i, t-1, :] # 掉包！卡在舊數值
    return X_dropped

# ================= 3. LTC-RNN 引擎 (使用 TensorFlow Keras Custom Layer 實作) =================
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

def build_ltc_models(num_neurons, input_shape=(TARGET_LINES, 5), num_classes=3):
    """
    建立兩個 Keras 模型，共用相同的權重：
    1. train_model: 只輸出最後一步的結果 (用於 BPTT 訓練算 CE Loss)
    2. interp_model: 輸出完整的 400 步時序軌跡 (用於畫圖與可解釋性分析)
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # LTC 循環層，設定 return_sequences=True 讓它保留每一刻的記憶
    ltc_cell = LTCNeuron(num_neurons)
    rnn_seq = tf.keras.layers.RNN(ltc_cell, return_sequences=True)(inputs)
    
    # 擷取最後一步的狀態
    rnn_final = rnn_seq[:, -1, :]
    
    # 最終的 Softmax 分類層
    dense_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
    final_output = dense_layer(rnn_final)
    
    # 1. 訓練用模型 (輸入 -> 最終機率)
    train_model = tf.keras.Model(inputs=inputs, outputs=final_output)
    
    # 2. 可解釋性模型 (輸入 -> [400步的隱藏狀態, 400步的 Softmax 機率])
    # 把同一個 Dense 層套用到整段序列上！
    seq_output = dense_layer(rnn_seq) 
    interp_model = tf.keras.Model(inputs=inputs, outputs=[rnn_seq, seq_output])
    
    return train_model, interp_model

# ================= 4. 訓練與掉封包盲測實驗 =================
if __name__ == "__main__":
    TARGET_NEURONS = 4 
    TRAIN_SAMPLES = 60 # 訓練固定使用 60 筆 (共 180 筆)
    
    drop_rates_to_test = [0.0, 0.2, 0.4, 0.6, 0.8]
    robustness_results = []

    print(f"\n{'='*70}")
    print(f"🔥 開始進行 LTC-4 掉封包實驗 (BPTT / TensorFlow Keras / Explicit Euler)")
    print(f"{'='*70}")

    print(f"\n🔹 [階段 1: 基礎模型訓練] 使用 {TRAIN_SAMPLES} 筆乾淨資料...")
    X_train_sub, Y_train_sub = get_few_shot_subset(X_train, Y_train, y_train_raw, TRAIN_SAMPLES)
    
    # 建立模型
    train_model, interp_model = build_ltc_models(TARGET_NEURONS, input_shape=(TARGET_LINES, 5), num_classes=3)
    
    # 設定優化器與 CE Loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    train_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], jit_compile=True)
    
    # 設定 EarlyStopping (自動儲存 Validation 最強的那一代！)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=50,               # 如果 40 個 Epoch 沒進步就停下來
        restore_best_weights=True, # 自動把權重倒帶回最強的那一刻
        verbose=1
    )
    
    # 開始 BPTT 訓練
    print("⏳ 開始 TensorFlow BPTT 訓練...")
    history = train_model.fit(
        X_train_sub, Y_train_sub,
        validation_data=(X_val, Y_val),
        epochs=500,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1 # 設定為 1 可看訓練進度條
    )

    print(f"\n🔹 [階段 2: 惡劣通訊環境盲測 (Zero-Order Hold)]")
    for drop_rate in drop_rates_to_test:
        X_test_dropped = apply_packet_loss(X_test, drop_rate)
        # 用 Keras 直接 Evaluate
        loss, test_acc = train_model.evaluate(X_test_dropped, Y_test, verbose=0)
        test_acc *= 100.0
        robustness_results.append(test_acc)
        print(f"   ▶ 掉包率 {drop_rate*100:2.0f}% | 盲測 Test Acc: {test_acc:.2f}%")

    # ================= 5. 繪製掉包容錯率折線圖 =================
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot([dr * 100 for dr in drop_rates_to_test], robustness_results, marker='o', linestyle='-', color='#e74c3c', linewidth=2.5, markersize=8)
    plt.title(f"LTC-4 Robustness vs. Packet Loss (TensorFlow BPTT)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Packet Drop Rate (%)", fontsize=12, fontweight='bold')
    plt.ylabel("Test Accuracy (%)", fontsize=12, fontweight='bold')
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    for i, txt in enumerate(robustness_results):
        plt.annotate(f"{txt:.1f}%", ([dr * 100 for dr in drop_rates_to_test][i], robustness_results[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    save_filename_line = "LTC4_PacketLoss_Robustness_TF.png"
    plt.savefig(save_filename_line, dpi=300)
    print(f"\n📊 折線圖已儲存為 {save_filename_line}")

    # ================= 6. 繪圖 (視覺化 50% 掉包情況下的動態推演) =================
    print("\n🔍 繪製 50% 掉封包極端情況下的內部狀態軌跡...")

    def plot_packet_loss_interpretability_tf(interp_model, X_clean, Y_true, num_neurons):
        time_steps = X_clean.shape[0]
        
        # 強制套用 50% 的掉包率
        X_dropped_batch = apply_packet_loss(np.expand_dims(X_clean, axis=0), drop_rate=0.5)
        X_dropped = X_dropped_batch[0] # 取出單筆資料畫圖用
        
        # 🌟 超級優雅：直接呼叫 Keras 的 interp_model，一次取得 400 步的 State 與 Probability！
        x_state_history, softmax_history = interp_model.predict(X_dropped_batch, verbose=0)
        x_state_history = x_state_history[0] # Shape: (400, 4)
        softmax_history = softmax_history[0] # Shape: (400, 3)

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, dpi=300)
        fig.suptitle("LTC-RNN Interpretability: Surviving 50% Packet Loss (TensorFlow)", fontsize=16, fontweight='bold', y=0.98)
        
        t_axis = np.arange(time_steps)
        
        # 1. 掉包後的輸入
        fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        colors_sensor = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']
        for i in range(5):
            axes[0].plot(t_axis, X_dropped[:, i], label=fingers[i], color=colors_sensor[i], linewidth=1.5, alpha=0.8)
        axes[0].set_title("1. Received Signals with 50% Packet Loss (Zero-Order Hold)", fontsize=12, fontweight='bold')
        axes[0].set_ylabel("Z-Score")
        axes[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # 2. Hidden Layer
        colors_neuron = ['#9b59b6', '#34495e', '#1abc9c', '#d35400']
        for n in range(num_neurons):
            axes[1].plot(t_axis, x_state_history[:, n], label=f'Neuron {n+1}', color=colors_neuron[n], linewidth=2.5)
        axes[1].set_title("2. Hidden Layer: LTC ODE States", fontsize=12, fontweight='bold')
        axes[1].set_ylabel("Activity State")
        axes[1].legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        axes[1].grid(True, linestyle='--', alpha=0.5)

        # 3. Softmax
        classes = ['Class 0', 'Class 1', 'Class 2']
        colors_prob = ['#7f8c8d', '#bdc3c7', '#95a5a6']
        true_label_idx = np.argmax(Y_true)
        colors_prob[true_label_idx] = '#c0392b' 
        
        for c in range(3):
            is_target = " (Target)" if c == true_label_idx else ""
            axes[2].plot(t_axis, softmax_history[:, c], label=f'{classes[c]} Prob{is_target}', color=colors_prob[c], linewidth=3 if c == true_label_idx else 1.5)
            
        axes[2].axvspan(time_steps - 10, time_steps, color='gray', alpha=0.3, label='Final Eval Point')
        
        axes[2].set_title("3. Output Layer: Probability Evolution", fontsize=12, fontweight='bold')
        axes[2].set_xlabel("Time Steps", fontsize=12, fontweight='bold')
        axes[2].set_ylabel("Probability")
        axes[2].set_ylim([-0.05, 1.05])
        axes[2].legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        axes[2].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        save_filename = "LTC_PacketLoss_Dynamics_TF.png"
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"📊 軌跡展示圖已儲存為 {save_filename}")

    sample_idx = 0
    plot_packet_loss_interpretability_tf(interp_model, X_test[sample_idx], Y_test[sample_idx], TARGET_NEURONS)