import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import tempfile

# 讓終端機繪圖暫存保持乾淨
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

# ================= 1. 全局設定 =================
TARGET_LINES = 400
# 🚨 請確認資料夾路徑是否正確
BASE_PATH = r'C:\Users\HAO\Desktop\YTY_from_macbook\dataset_xx2020_new_new_new\dataset_602020_zscore' 

# 定義要測試的所有模型名稱
# model_configs = ['1D-CNN', 'SimpleRNN-8', 'LSTM-8', 'LTC-4']
model_configs = ['LTC-4']

TRAINING_EPOCHS = 2000
BATCH_SIZE = 8
HISTORY_CSV_TEMPLATE = "history_{model_name}_epoch{epochs}.csv"
PLOT_TEMPLATE = "Training_Dynamics_{model_name}_Epoch{epochs}.png"


# 統一論文色系：橘、灰、紅、藍
model_palette = {
    '1D-CNN': '#ff7f0e', 
    'SimpleRNN-8': '#7f7f7f',
    'LSTM-8': '#d62728',
    'LTC-4': '#1f77b4'
}

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

# ================= 4. 主程式：訓練模型並記錄軌跡 =================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"🚀 開始執行模型訓練 (擷取單次 Training Dynamics)")
    print(f"{'='*60}")
    
    # --- A. 載入數據 ---
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

    # 用來記錄 CSV 檔案名稱的路徑字典
    history_files = {}

    # --- C. 依序訓練並記錄每個模型 ---
    for name in model_configs:
        print(f"\n🔹 [目前訓練架構：{name}]")

        # 依據架構名稱動態建立模型
        if name == '1D-CNN':
            model = keras.Sequential([
                keras.Input(shape=(TARGET_LINES, 5)),
                keras.layers.Conv1D(filters=15, kernel_size=3, strides=2, padding='valid', activation='relu'),
                keras.layers.Conv1D(filters=15, kernel_size=3, strides=2, padding='valid', activation='relu'),
                keras.layers.Conv1D(filters=15, kernel_size=3, strides=2, padding='valid', activation='relu'),
                keras.layers.Flatten(),
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

        # 編譯模型
        custom_adam = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0)
        model.compile(optimizer=custom_adam, loss='categorical_crossentropy', metrics=['accuracy'], jit_compile=True)

        # Fixed 2000-epoch BPTT curve: no early stopping.
        csv_file = HISTORY_CSV_TEMPLATE.format(
            model_name=name.lower().replace('-', ''),
            epochs=TRAINING_EPOCHS
        )
        history_files[name] = csv_file

        print(f"  ▶ 正在進行訓練並擷取 Loss/Accuracy 軌跡...", end=" ", flush=True)
        # 開始訓練
        history = model.fit(
            X_train_raw, y_train, 
            epochs=TRAINING_EPOCHS,
            batch_size=BATCH_SIZE, 
            validation_data=(X_val_raw, y_val),
            verbose=0 
        )

        history_df = pd.DataFrame(history.history)
        history_df.insert(0, 'epoch', np.arange(1, len(history_df) + 1))
        history_df.to_csv(csv_file, index=False)

        loss, accuracy = model.evaluate(X_test_raw, y_test, verbose=0)
        print(f"完成！Test Acc: {accuracy*100:>6.2f}%, 紀錄已存至 {csv_file}")


    # ================= 5. 繪製個別模型的 Training Dynamics 趨勢圖 =================
    print("\n📊 正在讀取 CSV 並繪製各模型的 Training Dynamics 趨勢圖...")

    for model_name, file_path in history_files.items():
        if not os.path.exists(file_path):
            print(f"⚠️ 找不到 {model_name} 的訓練紀錄 {file_path}，跳過繪製。")
            continue
            
        df = pd.read_csv(file_path)
        best_val_idx = df['val_loss'].idxmin()
        best_val_epoch = int(df.loc[best_val_idx, 'epoch'])
        best_val_loss = float(df.loc[best_val_idx, 'val_loss'])
        final_epoch = int(df['epoch'].iloc[-1])
        final_val_loss = float(df['val_loss'].iloc[-1])
        
        # 🔑 幫每個模型開一張新的畫布 (1x2 雙拼)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
        
        # 左圖：Loss
        axes[0].plot(df['epoch'], df['loss'], color=model_palette[model_name], linestyle='-', linewidth=2, alpha=0.8)
        axes[0].plot(df['epoch'], df['val_loss'], color='#d62728', linestyle='-', linewidth=2, alpha=0.8)
        axes[0].scatter(
            [best_val_epoch],
            [best_val_loss],
            color='#2ca02c',
            edgecolor='black',
            linewidth=0.6,
            s=55,
            zorder=5
        )
        axes[0].axvline(best_val_epoch, color='#2ca02c', linestyle=':', linewidth=1.4, alpha=0.9)
        axes[0].axvline(final_epoch, color='black', linestyle='--', linewidth=1.1, alpha=0.65)
        axes[0].annotate(
            f"Best val loss\nEpoch {best_val_epoch}\n{best_val_loss:.4f}",
            xy=(best_val_epoch, best_val_loss),
            xytext=(12, 18),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2ca02c', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='#2ca02c', linewidth=1.0)
        )
        axes[0].annotate(
            f"Final epoch {final_epoch}\nval loss {final_val_loss:.4f}",
            xy=(final_epoch, final_val_loss),
            xytext=(-115, -35),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.85),
            arrowprops=dict(arrowstyle='->', color='black', linewidth=0.9)
        )
        
        # 右圖：Accuracy (自動轉為百分比)
        train_acc = df['accuracy'] * 100 if df['accuracy'].max() <= 1.0 else df['accuracy']
        val_acc = df['val_accuracy'] * 100 if df['val_accuracy'].max() <= 1.0 else df['val_accuracy']
        
        axes[1].plot(df['epoch'], train_acc, color=model_palette[model_name], linestyle='-', linewidth=2, alpha=0.8)
        axes[1].plot(df['epoch'], val_acc, color='#d62728', linestyle='-', linewidth=2, alpha=0.8)
        axes[1].axvline(best_val_epoch, color='#2ca02c', linestyle=':', linewidth=1.4, alpha=0.9)
        axes[1].axvline(final_epoch, color='black', linestyle='--', linewidth=1.1, alpha=0.65)

        # 自訂圖例外觀 (直接使用該模型的專屬顏色)
        legend_elements = [
            Line2D([0], [0], color=model_palette[model_name], lw=2, linestyle='-', label=f'Training'),
            Line2D([0], [0], color='#d62728', lw=2, linestyle='-', label=f'Validation'),
            Line2D([0], [0], color='#2ca02c', lw=1.4, linestyle=':', label=f'Best val epoch ({best_val_epoch})'),
            Line2D([0], [0], color='black', lw=1.1, linestyle='--', label=f'Final epoch ({final_epoch})')
        ]

        # 左圖外觀設定
        axes[0].set_title(f"{model_name} - Loss over {final_epoch} Epochs", fontsize=16, fontweight='bold', pad=15)
        axes[0].set_xlabel("Epoch", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("Loss (Categorical Crossentropy)", fontsize=14, fontweight='bold')
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)

        # 右圖外觀設定
        axes[1].set_title(f"{model_name} - Accuracy over {final_epoch} Epochs", fontsize=16, fontweight='bold', pad=15)
        axes[1].set_xlabel("Epoch", fontsize=14, fontweight='bold')
        axes[1].set_ylabel("Accuracy (%)", fontsize=14, fontweight='bold')
        axes[1].grid(True, linestyle='--', alpha=0.6)
        axes[1].set_ylim([0, 100]) 

        # 儲存獨立圖檔
        plt.tight_layout()
        save_name = PLOT_TEMPLATE.format(
            model_name=model_name,
            epochs=final_epoch
        )
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"🎉 {model_name} 趨勢圖已成功儲存為 {save_name}")
    
    # 一次顯示剛畫好的 4 張圖
    plt.show()
