import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import sys
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from dataset_loader_3ch import DEFAULT_DATASET_PATH, load_split_dataset

# ================= 全局設定 =================
TARGET_LINES = 400
BASE_PATH = DEFAULT_DATASET_PATH

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
        features = df[['Thumb', 'Middle', 'Pinky']].values

        if features.shape == (TARGET_LINES, 3):
            signal_list.append(features)
            label_list.append(label)
            
    return np.array(signal_list), np.array(label_list)

def load_split_data(base_path):
    print("🔍 正在載入新的 3-channel dataset...")
    (signal_train_raw, label_train_raw), (signal_val_raw, label_val_raw), (signal_test_raw, label_test_raw) = load_split_dataset(base_path)

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

# ================= 2. 定義 4 顆 LTC 神經元 =================
class LTCNeuron(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(LTCNeuron, self).__init__(**kwargs)
        self.units = units
        self.state_size = units  

    def build(self, input_shape):
        input_dim = input_shape[-1] # 這裡是 5 (手指數)
        init_w = keras.initializers.RandomNormal(stddev=0.5) 
        init_ones = 'ones'
        init_zeros = 'zeros'

        # 陣列為二維矩陣 (5 根手指, 4 顆神經元)
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
        
        # 陽春版：顯式歐拉更新 (容易因數值不穩定而過衝)
        dx = -damping * x + driving
        x_new = x + (delta_t * dx)
        
        return x_new, [x_new]

# ================= 3. 主程式執行區塊 =================
if __name__ == "__main__":
    # --- A. 準備數據 ---
    (train_raw, y_train_raw), (val_raw, y_val_raw), (test_raw, y_test_raw) = load_split_data(BASE_PATH)
    y_train, y_val, y_test = encode_label(y_train_raw, y_val_raw, y_test_raw)
    X_train_scaled, X_val_scaled, X_test_scaled = train_raw, val_raw, test_raw


    # --- B. 建立與編譯模型 ---
    model = keras.Sequential([
        keras.Input(shape=(TARGET_LINES, 3)), 
        # 🌟 呼叫 4 顆神經元的版本 (Units = 4)
        keras.layers.RNN(LTCNeuron(units=4), return_sequences=False), 
        keras.layers.Dense(10, activation='softmax')
    ])

    custom_adam = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0)
    model.compile(optimizer=custom_adam, loss='categorical_crossentropy', metrics=['accuracy'], jit_compile=True)

    # --- C. 開始訓練 ---
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=200, restore_best_weights=True, verbose=1
    )
    print("\n🚀 開始訓練 4 顆 LTC 神經元模型 (LTC-4)...")
    history = model.fit(
        X_train_scaled, y_train, 
        epochs=2000,
        batch_size=8, 
        validation_data=(X_val_scaled, y_val),
        callbacks=[early_stopping],
        verbose=1 
    )
    
    # --- D. 最終驗證 ---
    print("\n📝 正在對 Test Set 進行最終驗證...")
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"🏆 最終 Test Accuracy: {accuracy*100:.2f}%\n")

    # --- E. 繪製並儲存圖表 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.suptitle('LTCRNN_4neuron_BPTT Training Results', fontsize=18, fontweight='bold', y=1.02)

    color_train = '#C46D4B'  
    color_val = '#5FAAA0'    

    # Accuracy
    ax1.plot(history.history['accuracy'], color=color_train, linewidth=2.5, label='Train')
    ax1.plot(history.history['val_accuracy'], color=color_val, linewidth=2.5, label='Val')
    ax1.set_xlabel('Epochs', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.set_title('a) Model Accuracy', fontsize=14, pad=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right')
    ax1.set_ylim([0.0, 1.0])

    # Loss
    ax2.plot(history.history['loss'], color=color_train, linewidth=2.5, label='Train')
    ax2.plot(history.history['val_loss'], color=color_val, linewidth=2.5, label='Val')
    ax2.set_xlabel('Epochs', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=14)
    ax2.set_title('b) Model Loss', fontsize=14, pad=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')
    ax2.set_ylim([0.0, 1.2])
    
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    fig.text(0.5, 0.03, f'Final Test Accuracy: {accuracy*100:.2f}%', 
             ha='center', va='center', fontsize=16, fontweight='bold', color='#333333',
             bbox=dict(facecolor='#F5F5F5', edgecolor='#CCCCCC', boxstyle='round,pad=0.5'))

    plt.savefig('LTCRNN_4neuron_BPTT_Training_Results.png', dpi=300, bbox_inches='tight')
    print("📊 訓練曲線已儲存為 LTCRNN_4neuron_BPTT_Training_Results.png")




    print("\n" + "="*80)
    print("🎯 BPTT 訓練完畢！各參數板塊的最佳範圍 (四捨五入至整數)：")
    print("="*80)

    ltc_cell = model.layers[0].cell
    w_np = ltc_cell.w.numpy()
    r_np = ltc_cell.r.numpy()
    mu_np = ltc_cell.mu.numpy()
    dense_w_np, dense_b_np = model.layers[1].get_weights()

    # 印出整數範圍
    print(f"w       range: {int(np.round(np.min(w_np))):>4d}  to  {int(np.round(np.max(w_np))):>4d}")
    print(f"r       range: {int(np.round(np.min(r_np))):>4d}  to  {int(np.round(np.max(r_np))):>4d}")
    print(f"mu      range: {int(np.round(np.min(mu_np))):>4d}  to  {int(np.round(np.max(mu_np))):>4d}")
    print(f"dense_W range: {int(np.round(np.min(dense_w_np))):>4d}  to  {int(np.round(np.max(dense_w_np))):>4d}")
    print(f"dense_b range: {int(np.round(np.min(dense_b_np))):>4d}  to  {int(np.round(np.max(dense_b_np))):>4d}")
    
    print("\n" + "="*80)
    print("🐍 Python 陣列格式 (純整數，可直接複製貼上)：")
    print("="*80)

    # 將矩陣展平、四捨五入，並強制轉型為 int 整數，再轉成 list
    w_list = np.round(w_np.flatten()).astype(int).tolist()
    r_list = np.round(r_np.flatten()).astype(int).tolist()
    mu_list = np.round(mu_np.flatten()).astype(int).tolist()
    dense_w_list = np.round(dense_w_np.flatten()).astype(int).tolist()
    dense_b_list = np.round(dense_b_np.flatten()).astype(int).tolist()

    print("best_w = np.array(", w_list, ")\n")
    print("best_r = np.array(", r_list, ")\n")
    print("best_mu = np.array(", mu_list, ")\n")
    print("best_dense_W = np.array(", dense_w_list, ")\n")
    print("best_dense_b = np.array(", dense_b_list, ")\n")

    # 🌟 直接幫你組裝好「終極殺招」的 110 參數一維陣列
    all_weights_1d = w_list + r_list + mu_list + dense_w_list + dense_b_list
    
    print("="*80)
    print("🚀 [終極殺招專用] 110 個參數的一維完整陣列 (純整數乾淨版)")
    print("請將以下這整行保存為模型參數匯出參考：")
    print("="*80)
    print(f"best_bptt_weights_1d = np.array({all_weights_1d})")
