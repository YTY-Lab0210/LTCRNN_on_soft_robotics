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

# 讓終端機保持乾淨，消除不必要的 TensorFlow 警告
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ================= 1. 全局設定 =================
TARGET_LINES = 400

BASE_PATH = r'C:\Users\HAO\Desktop\YTY_from_macbook\dataset_xx2020_new_new_new\dataset_602020_zscore'

# 測試的神經元數量
unit_configs = [1, 2, 4, 8, 16]
num_runs = 10
all_accuracies = []

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

# ================= 4. 主程式：神經元消融實驗 =================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"🚀 開始執行 LTC 神經元消融實驗 (Ablation Study)")
    print(f"{'='*60}")
    
    # --- A. 載入數據 (只需載入一次) ---
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

    # --- D. 外層迴圈：切換神經元數量 ---
    for u in unit_configs:
        print(f"\n🧠 [目前測試架構：{u} 顆 LTC 神經元]")
        current_neuron_accuracies = []

        # --- E. 內層迴圈：重複訓練 10 次 ---
        for run in range(num_runs):
            print(f"  ▶ 開始第 {run + 1:02d}/{num_runs} 次訓練...", end=" ", flush=True)
            
            model = keras.Sequential([
                keras.Input(shape=(TARGET_LINES, 5)), 
                keras.layers.RNN(LTCNeuron(units=u), return_sequences=False), 
                keras.layers.Dense(10, activation='softmax')
            ])

            custom_adam = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0)
            model.compile(optimizer=custom_adam, loss='categorical_crossentropy', metrics=['accuracy'], jit_compile=True)

            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=200, restore_best_weights=True, verbose=0 # 設為 0 保持終端機乾淨
            )

            history = model.fit(
                X_train_raw, y_train,
                epochs=2000,
                batch_size=8, 
                validation_data=(X_val_raw, y_val),
                callbacks=[early_stopping],
                verbose=0
            )

            # 🌟 測試時同樣使用高效流水線
            loss, accuracy = model.evaluate(X_test_raw, y_test, verbose=0)
            current_neuron_accuracies.append(accuracy* 100)
            print(f"完成！Test Acc: {accuracy*100:>6.2f}%")
            
        all_accuracies.append(current_neuron_accuracies)
        print(f"📊 {u} 顆神經元 10次平均準確率: {np.mean(current_neuron_accuracies):.2f}%\n")


    # ================= 5. 繪製並儲存神級 Boxplot 盒鬚圖 =================
    if len(all_accuracies) > 0:
        print("\n🎨 正在繪製並儲存 Boxplot 盒鬚圖...")

        all_accuracies_percent = [[val for val in exp] for exp in all_accuracies]
        plot_labels = [f"LTC-{u}" for u in unit_configs] 

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        bplot = ax.boxplot(all_accuracies_percent, tick_labels=plot_labels, patch_artist=True, zorder=1)

        colors = ['#ccebc5', '#a8ddb5', '#7bccc4', '#4eb3d3', '#2b8cbe']

        for i, patch in enumerate(bplot['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.8)
            patch.set_edgecolor('#333333')
            patch.set_linewidth(1.5)

        for median_line in bplot['medians']:
            median_line.set(color='#d73027', linewidth=2.5) 

        for whisker in bplot['whiskers']:
            whisker.set(linewidth=1.5, linestyle='--')
        for cap in bplot['caps']:
            cap.set(linewidth=1.5)

        for i, data_pts in enumerate(all_accuracies_percent):
            x_jitter = np.random.normal(i + 1, 0.04, size=len(data_pts))
            ax.scatter(x_jitter, data_pts, color='black', alpha=0.6, s=30, edgecolor='white', linewidth=0.8, zorder=2)

        global_min = float('inf')
        for i, data_pts in enumerate(all_accuracies_percent):
            mean_val = np.mean(data_pts)        
            min_val = np.min(data_pts)          
            center_x = i + 1  
            
            if min_val < global_min:
                global_min = min_val
                
            ax.text(center_x, min_val - 2.0, f'Mean:\n{mean_val:.1f}%', 
                     ha='center', va='top', fontsize=11, fontweight='bold', color='black',
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='#CCCCCC', boxstyle='round,pad=0.3'), zorder=4)

        ax.set_title('Ablation Study: LTC Neuron Capacity vs. Performance (10 Runs)', fontsize=15, fontweight='bold', pad=15)
        ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model Architecture', fontsize=14, fontweight='bold')

        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0) 
        ax.set_ylim([max(0, global_min - 10), 100]) 

        plt.tight_layout()
        save_name = 'LTCRNN_Ablation_Boxplot.png'
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"✅ Boxplot 已成功儲存為 {save_name}！")
        
        # ================= 6. 儲存原始實驗數據 (CSV) =================
        print("💾 正在將實驗結果匯出為 CSV 檔案...")
        flat_results = []
        for i, u in enumerate(unit_configs):
            for run_idx, acc in enumerate(all_accuracies[i]):
                flat_results.append({
                    'Architecture': f"LTC-{u}",
                    'Run': run_idx + 1,
                    'Accuracy(%)': acc
                })
                
        df_results = pd.DataFrame(flat_results)
        csv_filename = "LTCRNN_Ablation_Raw_Data.csv"
        df_results.to_csv(csv_filename, index=False)
        print(f"💾 原始測試數據已安全儲存為 {csv_filename}！")

        plt.show()