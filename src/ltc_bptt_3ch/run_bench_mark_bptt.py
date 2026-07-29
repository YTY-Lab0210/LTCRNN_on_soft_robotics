import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
from dataset_loader_3ch import DEFAULT_DATASET_PATH, load_split_dataset

# ================= 1. 全局設定 =================
TARGET_LINES = 400
# 🚨 請確認資料夾路徑是否正確
BASE_PATH = DEFAULT_DATASET_PATH

# 定義要測試的所有模型名稱
model_configs = ['1D-CNN', 'SimpleRNN-8', 'LSTM-8', 'LTC-4']

num_runs = 10

# 使用字典儲存所有對照組的成績
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
        features = df[['Thumb', 'Middle', 'Pinky']].values

        if features.shape == (TARGET_LINES, 3):
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
    print(f"🚀 開始執行多模型架構基準測試 (BPTT 大亂鬥)")
    print(f"{'='*60}")
    
    # --- A. 載入數據 (只需載入一次) ---
    (X_train_raw, y_train_raw), (X_val_raw, y_val_raw), (X_test_raw, y_test_raw) = load_split_dataset(BASE_PATH)
    
    # --- B. 標籤轉換 ---
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_raw)
    y_val_encoded   = encoder.transform(y_val_raw)
    y_test_encoded  = encoder.transform(y_test_raw)

    num_classes = len(encoder.classes_)
    y_train = to_categorical(y_train_encoded, num_classes)
    y_val   = to_categorical(y_val_encoded, num_classes)
    y_test  = to_categorical(y_test_encoded, num_classes)

    print(f"類別有: {num_classes}\n")


    # output_num = 8

    # --- D. 外層迴圈：依序切換不同的模型架構 ---
    for name in model_configs:
        print(f"\n🔹 [目前測試架構：{name}]")

        # --- E. 內層迴圈：重複訓練 10 次 ---
        for run in range(num_runs):
            print(f"  ▶ 開始第 {run + 1:02d}/{num_runs} 次訓練...", end=" ", flush=True)
            
            # 依據架構名稱動態建立模型
            if name == '1D-CNN':
                model = keras.Sequential([
                    keras.Input(shape=(TARGET_LINES, 3)),
                    keras.layers.Conv1D(filters=15, kernel_size=3, strides=2, padding='valid', activation='relu'),
                    keras.layers.Conv1D(filters=15, kernel_size=3, strides=2, padding='valid', activation='relu'),
                    keras.layers.Conv1D(filters=15, kernel_size=3, strides=2, padding='valid', activation='relu'),
                    keras.layers.Flatten(),
                    keras.layers.Dense(num_classes, activation='softmax')
                ]) 

            elif name == 'SimpleRNN-8':
                model = keras.Sequential([
                    keras.Input(shape=(TARGET_LINES, 3)),
                    keras.layers.SimpleRNN(8, return_sequences=False),
                    keras.layers.Dense(num_classes, activation='softmax')
                ])

            elif name == 'LSTM-8':
                model = keras.Sequential([
                    keras.Input(shape=(TARGET_LINES, 3)),
                    keras.layers.LSTM(8, return_sequences=False),
                    keras.layers.Dense(num_classes, activation='softmax')
                ])
            elif name == 'LTC-4':
                model = keras.Sequential([
                    keras.Input(shape=(TARGET_LINES, 3)),
                    keras.layers.RNN(LTCNeuron(units=4), return_sequences=False),
                    keras.layers.Dense(num_classes, activation='softmax')
                ])
            elif name == 'LTC-8':
                model = keras.Sequential([
                    keras.Input(shape=(TARGET_LINES, 3)),
                    keras.layers.RNN(LTCNeuron(units=8), return_sequences=False),
                    keras.layers.Dense(num_classes, activation='softmax')
                ])

            # 🔑 統一使用與標準程式碼完全相同的優化器配置
            custom_adam = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0)
            model.compile(optimizer=custom_adam, loss='categorical_crossentropy', metrics=['accuracy'], jit_compile=True)

            if(run == 0):
                model.summary()

            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=200, restore_best_weights=True, verbose=1
            )

            # 🔑 統一使用 500 epochs, batch_size=8
            history = model.fit(
                X_train_raw, y_train, 
                epochs=2000,
                batch_size=8, 
                validation_data=(X_val_raw, y_val),
                callbacks=[early_stopping],
                verbose=0 
            )

            loss, accuracy = model.evaluate(X_test_raw, y_test, verbose=0)
            
            # 存入字典中 (轉換為百分比)
            all_test_accuracies[name].append(accuracy * 100)
            print(f"完成！Test Acc: {accuracy*100:>6.2f}%")
            
        print(f"📊 {name} 10次平均準確率: {np.mean(all_test_accuracies[name]):.2f}%\n")


    # ================= 將結果存檔 (防呆機制) =================
    df_results = pd.DataFrame(all_test_accuracies)
    df_results.to_csv("Benchmark_Results_BPTT.csv", index=False)
    print("\n✅ 所有結果已成功儲存至 Benchmark_Results_BPTT.csv")

    # ================= 繪製精美 Boxplot =================
    print("\n📊 正在繪製多模型基準測試 Boxplot...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300) # 比例調為 10:6 視覺上最舒適
    
    # 提取要畫的數據陣列
    data_to_plot = [all_test_accuracies[name] for name in model_configs]
    labels = model_configs
    
    # 配好 6 個模型的漂亮精緻色系 (對應你可能增加的 LTC 變體)
    model_palette = ['#ff7f0e', '#7f7f7f', '#d62728', '#1f77b4']
    
    # 畫箱型圖
    bplot = ax.boxplot(data_to_plot, patch_artist=True, labels=labels, 
                       widths=0.5, showmeans=True)

    # 幫箱子上色與外觀設定
    for patch, color in zip(bplot['boxes'], model_palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for median in bplot['medians']:
        median.set(color='black', linewidth=2)

    # 🌟 疊加個別的 10 次實驗資料點 (Scatter plot with Jitter)
    for i, data_pts in enumerate(data_to_plot):
        x_jitter = np.random.normal(i + 1, 0.04, size=len(data_pts))
        plt.scatter(x_jitter, data_pts, color='black', alpha=0.6, s=30, edgecolor='white', linewidth=0.8, zorder=2)

    # 🌟 計算平均值，並動態把標籤放在該組數據的「最低分下方」
    global_min = float('inf')
    for i, data_pts in enumerate(data_to_plot):
        mean_val = np.mean(data_pts)        
        min_val = np.min(data_pts)          
        center_x = i + 1  
        
        if min_val < global_min:
            global_min = min_val
            
        # 將文字放在該組數據最低分的下方 (min_val - 2.0)
        plt.text(center_x, min_val - 2.0, f'Mean: {mean_val:.1f}%', 
                 ha='center', va='top', fontsize=11, fontweight='bold', color='black',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=1'), zorder=4)

    # 設定標題與軸標籤
    ax.set_title(r"Model Architecture Performance Comparison (10 Runs) | BPTT", fontsize=15, fontweight='bold', pad=15)
    ax.set_ylabel("Test Accuracy (%)", fontsize=14, fontweight='bold')
    
    # 動態調整 Y 軸：確保最底下的標籤不會被切掉，上限固定到 105 留出視覺空間
    ax.set_ylim([max(0, global_min - 10), 100])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("Benchmark_Accuracy_Boxplot_BPTT.png", dpi=300, bbox_inches='tight')
    print("📊 完美！基準測試 Boxplot 圖表已成功儲存為 Benchmark_Accuracy_Boxplot_BPTT.png")
    
    plt.show()
