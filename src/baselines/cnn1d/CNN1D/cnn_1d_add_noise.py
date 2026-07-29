import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Input
from keras.utils import to_categorical

# ================= 1. 讀取與塑形 =================
dataset_path = '/Users/laihao/Desktop/YTY/dataset'
# all_files = glob.glob(os.path.join(dataset_path, "*.csv"))
all_files = sorted(glob.glob(os.path.join(dataset_path, "*.csv")))


X_list = []
y_list = []
TARGET_LINES = 400

print(f"📂 目前設定的讀取路徑為: {dataset_path}")
print(f"📊 實際找到的 CSV 檔案數量: {len(all_files)}")
if len(all_files) == 0:
    print("❌ 錯誤：找不到 CSV 檔案！請確認路徑。")
    exit()

print("🔍 開始掃描並載入檔案...")
for file in all_files:
    filename = os.path.basename(file)
    label = filename.rsplit('_', 1)[0]

    df = pd.read_csv(file)
    features = df[['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']].values

    if features.shape == (TARGET_LINES, 5):
        X_list.append(features)
        y_list.append(label)

X = np.array(X_list)
y = np.array(y_list)

# ================= 1.5 數據標準化 =================
num_samples, timesteps, num_features = X.shape
X_2d = X.reshape(-1, num_features)
scaler = StandardScaler()
X_2d_scaled = scaler.fit_transform(X_2d)
X_scaled = X_2d_scaled.reshape(num_samples, timesteps, num_features)

# ================= 2. 標籤轉換 =================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
y_categorical = to_categorical(y_encoded, num_classes)

# ================= 3. 切分訓練集與測試集 =================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded)
print(f"\n📝 原始切分 -> 訓練集: {X_train.shape[0]} 組，測試集: {X_test.shape[0]} 組")


# ================= 🌟 3.5 資料擴增 (Data Augmentation) 🌟 =================
print("🧬 開始進行資料擴增 (注入 Gaussian Noise)...")

# 設定雜訊強度 (0.05 模擬氣動手臂的微小震動)
noise_factor = 0.05

# 產生與原始 X_train 形狀一模一樣的隨機雜訊矩陣
noise = np.random.normal(loc=0.0, scale=noise_factor, size=X_train.shape)

# 將雜訊疊加到原始訓練資料上，創造出帶有抖動的新版本
X_train_noisy = X_train + noise

# 將原本的乾淨資料與新的雜訊資料「上下疊合」起來 (48 + 48 = 96)
X_train_aug = np.concatenate((X_train, X_train_noisy), axis=0)

# 標籤也要跟著複製疊合 (確保對應正確)
y_train_aug = np.concatenate((y_train, y_train), axis=0)

# ⚠️ 關鍵：打亂順序 (Shuffle)
# 如果不打亂，模型會連續學完乾淨的再學髒的，效果不好
indices = np.arange(X_train_aug.shape[0])
np.random.shuffle(indices)

# 將擴增且打亂後的資料，正式指派給 X_train 與 y_train
X_train = X_train_aug[indices]
y_train = y_train_aug[indices]

print(f"🌳 擴增完成！目前訓練集暴增為: {X_train.shape[0]} 組 (已完美打亂)")
# ===========================================================================


# ================= 4. 建立老師指定的 MNIST-1D 架構 =================
model = Sequential()
model.add(Input(shape=(TARGET_LINES, 5)))
model.add(Conv1D(filters=15, kernel_size=3, strides=2, activation='relu'))
model.add(Conv1D(filters=15, kernel_size=3, strides=2, activation='relu'))
model.add(Conv1D(filters=15, kernel_size=3, strides=2, activation='relu'))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ================= 5. 開始訓練 =================
print("\n🔥 使用擴增後的資料開始訓練...")
# 注意：這裡依然是拿乾淨的 X_test 來做期末考
history = model.fit(X_train, y_train, epochs=40, batch_size=8, validation_data=(X_test, y_test))

# ================= 6. 期末考驗證 =================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🏆 最終測試集辨識準確率 (Test Accuracy): {accuracy*100:.2f}%")