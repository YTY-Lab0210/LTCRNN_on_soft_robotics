import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Input
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping # 🌟 1. 匯入 EarlyStopping 模組
import matplotlib.pyplot as plt

TARGET_LINES = 400

# ================= 1. 定義讀取資料夾的專屬函式 =================
def load_sensor_data(folder_path):
    """從指定資料夾讀取所有 CSV 並回傳特徵 X 與標籤 y"""
    all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    X_list = []
    y_list = []
    
    if len(all_files) == 0:
        print(f"⚠️ 警告：在 {folder_path} 中找不到任何 CSV 檔案！")
        return np.array([]), np.array([])

    for file in all_files:
        filename = os.path.basename(file)
        label = filename.rsplit('_', 1)[0]

        df = pd.read_csv(file)
        features = df[['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']].values

        if features.shape == (TARGET_LINES, 5):
            X_list.append(features)
            y_list.append(label)
            
    return np.array(X_list), np.array(y_list)

# ================= 2. 分別載入三個資料夾的數據 =================
base_path = '/Users/laihao/Desktop/YTY/dataset_split' # 確保這裡是你的新資料夾路徑

print("🔍 正在從三個資料夾獨立載入數據...")
X_train_raw, y_train_raw = load_sensor_data(os.path.join(base_path, 'training'))
X_val_raw, y_val_raw     = load_sensor_data(os.path.join(base_path, 'validation'))
X_test_raw, y_test_raw   = load_sensor_data(os.path.join(base_path, 'test'))

print(f"✅ 載入完成！ Train: {len(X_train_raw)}筆 | Val: {len(X_val_raw)}筆 | Test: {len(X_test_raw)}筆")

# ================= 3. 標籤轉換 (Label Encoding) =================
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train_raw)
y_val_encoded   = encoder.transform(y_val_raw)
y_test_encoded  = encoder.transform(y_test_raw)

num_classes = len(encoder.classes_)
y_train = to_categorical(y_train_encoded, num_classes)
y_val   = to_categorical(y_val_encoded, num_classes)
y_test  = to_categorical(y_test_encoded, num_classes)

# ================= 4. 數據標準化 =================
num_features = 5
scaler = StandardScaler()

X_train_2d = X_train_raw.reshape(-1, num_features)
X_val_2d   = X_val_raw.reshape(-1, num_features)
X_test_2d  = X_test_raw.reshape(-1, num_features)

X_train_scaled = scaler.fit_transform(X_train_2d).reshape(-1, TARGET_LINES, num_features)
X_val_scaled   = scaler.transform(X_val_2d).reshape(-1, TARGET_LINES, num_features)
X_test_scaled  = scaler.transform(X_test_2d).reshape(-1, TARGET_LINES, num_features)

# ... (前面 1 到 4 步驟的讀取與標準化程式碼完全保留) ...

# ================= 🌟 5. 將模型架構打包成函式 (為了確保每次起跑點公平) =================
def build_model():
    model = Sequential()
    model.add(Input(shape=(TARGET_LINES, 5)))
    model.add(Conv1D(filters=15, kernel_size=3, strides=2, activation='relu'))
    model.add(Conv1D(filters=15, kernel_size=3, strides=2, activation='relu'))
    model.add(Conv1D(filters=15, kernel_size=3, strides=2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ================= 6. 進行「單一回合」完整訓練 =================
EPOCHS = 100
print(f"\n🚀 開始單回合訓練：強迫跑完 {EPOCHS} 代 (萃取完整歷史紀錄)...")

model = build_model()
# 這裡刻意不放 callbacks，讓它完整跑完
history = model.fit(
    X_train_scaled, y_train, 
    epochs=EPOCHS, 
    batch_size=8, 
    validation_data=(X_val_scaled, y_val),
    verbose=0
)

# --- 🌟 神級操作：用程式碼回推 Early Stopping 的觸發點 ---
val_losses = history.history['val_loss']
patience = 15
best_epoch = 0
best_val_loss = val_losses[0]
stopped_epoch = EPOCHS - 1 # 預設沒停的話就是最後一代

for i in range(len(val_losses)):
    # 如果找到更低的 loss，更新最佳紀錄
    if val_losses[i] < best_val_loss:
        best_val_loss = val_losses[i]
        best_epoch = i
        
    # 如果已經連續 patience 代沒有進步，觸發假想的 Early Stopping
    if i - best_epoch >= patience:
        stopped_epoch = i
        break

print(f"✅ 分析完成！")
print(f"   - 最佳大腦狀態發生在：第 {best_epoch} 代")
print(f"   - 若開啟 Early Stopping，系統會在：第 {stopped_epoch} 代自動煞車")


# ================= 7. 繪製終極對比圖 (完美同起跑點，精準切斷 x 軸) =================
print("\n📊 正在繪製 Loss 專屬對比圖...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
fig.suptitle('Validation Loss Comparison (Same Training Run)', fontsize=18, fontweight='bold', y=1.05)

color_train = '#C46D4B'  # 橘色 (Train Loss)
color_es = '#5CB85C'     # 綠色 (有煞車)
color_no_es = '#D9534F'  # 紅色 (無煞車)

# ----------------- 左圖：【有】 Early Stopping (精準切斷 x 軸) -----------------
# 切片資料到 stopped_epoch
ax1.plot(history.history['loss'][:stopped_epoch+1], color=color_train, linewidth=2, linestyle='-', alpha=0.7, label='Train Loss')
ax1.plot(history.history['val_loss'][:stopped_epoch+1], color=color_es, linewidth=3, label='Val Loss')

# 畫一條虛線標示最佳權重點
# ax1.axvline(x=best_epoch, color='gray', linestyle=':', linewidth=2)
# ax1.text(best_epoch+1, ax1.get_ylim()[1]*0.8, f'Best Weights\n(Epoch {best_epoch})', color='gray', fontweight='bold')

ax1.set_xlabel('Epochs', fontsize=14)
ax1.set_ylabel('Categorical Crossentropy (Loss)', fontsize=14)
ax1.set_title(f'a) With Early Stopping', fontsize=14, pad=10)

# 🌟 關鍵微調 1：強制將左圖的 X 軸視野鎖死在「停止的那一代」
# ax1.set_xlim(0, stopped_epoch + 1)
ax1.set_xlim(0, best_epoch + 1)


ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(fontsize=12)

# ----------------- 右圖：【無】 Early Stopping (畫出完整 100 代) -----------------
ax2.plot(history.history['loss'], color=color_train, linewidth=2, linestyle='-', alpha=0.7, label='Train Loss')
ax2.plot(history.history['val_loss'], color=color_no_es, linewidth=3, label='Val Loss')

# 標示原本應該煞車的地方
ax2.axvline(x=stopped_epoch, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
ax2.text(stopped_epoch-15, ax2.get_ylim()[1]*0.6, 'Early Stopping\nTrigger Point', color='black', fontsize=10, alpha=0.7)

ax2.axvline(x=best_epoch, color='gray', linestyle=':', linewidth=2)
ax2.text(best_epoch+1, ax2.get_ylim()[1]*0.8, f'Best Weights\n(Epoch {best_epoch})', color='gray', fontweight='bold')

# 抓取右圖最後階段的點，畫一個箭頭指出過擬合反彈
val_loss_end = history.history['val_loss'][-5] # 取倒數第 5 代當作標示點
# ax2.annotate('Overfitting\n(Val Loss Rebounds)', 
#              xy=(EPOCHS-5, val_loss_end), 
#              xytext=(EPOCHS-40, val_loss_end + 0.3),
#              arrowprops=dict(facecolor='#D9534F', shrink=0.05, width=2, headwidth=8),
#              fontsize=12, color='#D9534F', fontweight='bold')

ax2.set_xlabel('Epochs', fontsize=14)
ax2.set_ylabel('Categorical Crossentropy (Loss)', fontsize=14)
ax2.set_title(f'b) Without Early Stopping', fontsize=14, pad=10)

# 🌟 關鍵微調 2：強制將右圖的 X 軸視野撐開到完整的 100 代
ax2.set_xlim(0, stopped_epoch)

ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(fontsize=12)

plt.tight_layout()
plt.savefig('loss_comparison_same_run_fixed_axis.png', dpi=300, bbox_inches='tight')
print("✅ 完美鎖定座標軸的對比圖已儲存為 loss_comparison_same_run_fixed_axis.png！")