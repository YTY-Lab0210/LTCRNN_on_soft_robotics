import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ================= 參數設定 =================
TARGET_LINES = 400
# 你的原始測試集路徑 (請依實際情況修改)
BASE_TEST_PATH = r'C:\Users\HAO\Desktop\YTY_from_macbook\dataset_xx2020_new\dataset_602020_zscore\test'
# 想要變慢的倍率 (1.5代表變慢 1.5 倍)
SLOW_FACTOR = 3
# 輸出的新測試集路徑
OUTPUT_PATH = f"{BASE_TEST_PATH}_slow_{SLOW_FACTOR}x"

os.makedirs(OUTPUT_PATH, exist_ok=True)

def stretch_and_crop_signal(features, slow_factor):
    """
    利用插值法將 400 步的訊號拉長，並使用「置中裁切 (Centered Crop)」
    確保抓握的過渡期與穩態期完整保留在 400 步的視窗內。
    """
    original_steps = len(features) # 400
    stretched_steps = int(original_steps * slow_factor) # 例如 1.5x 會變成 600
    
    # 建立原本的時間軸與拉長後的時間軸
    t_orig = np.linspace(0, 1, original_steps)
    t_stretched = np.linspace(0, 1, stretched_steps)
    
    # 對 5 個感測器通道分別進行線性插值
    interpolator = interp1d(t_orig, features, axis=0, kind='linear', fill_value='extrapolate')
    stretched_features = interpolator(t_stretched)
    
    # 🌟 核心修正：動態計算置中裁切的起始與結束點
    # 例如 stretch_steps=600, target=400 ➔ start_idx = (600-400)//2 = 100
    # 正好符合你說的 100 ~ 500！
    start_idx = (stretched_steps - TARGET_LINES) // 2
    end_idx = start_idx + TARGET_LINES
    
    # 截取中間最精華的 400 步
    final_features = stretched_features[start_idx:end_idx, :]
    
    return final_features

# ================= 執行轉換 =================
all_files = sorted(glob.glob(os.path.join(BASE_TEST_PATH, "*.csv")))
print(f"🔍 找到 {len(all_files)} 個測試檔案，準備生成變慢 {SLOW_FACTOR} 倍的測試集...")

for i, file in enumerate(all_files):
    filename = os.path.basename(file)
    df = pd.read_csv(file)
    
    # 提取 5 指數值
    features = df[['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']].values
    
    # 產生變慢的訊號
    slow_features = stretch_and_crop_signal(features, SLOW_FACTOR)
    
    # 存成新的 CSV
    df_slow = pd.DataFrame(slow_features, columns=['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'])
    save_file = os.path.join(OUTPUT_PATH, filename)
    df_slow.to_csv(save_file, index=False)
    
    # 畫出第一筆資料的對照圖，讓你視覺化確認物理效果
    if i == 0:
        plt.figure(figsize=(10, 4), dpi=150)
        plt.plot(features[:, 0], label='Original (Thumb)', color='gray', linestyle='--')
        plt.plot(slow_features[:, 0], label=f'Slowed {SLOW_FACTOR}x (Thumb)', color='red', linewidth=2)
        plt.title(f"Time-Warping Effect on Sensor Data ({filename})")
        plt.xlabel("Time Steps")
        plt.ylabel("Sensor Value (Z-score)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('Slow_Effect_Demo_3.0x.png', bbox_inches='tight')
        plt.close()

print(f"\n✅ 全部轉換完成！變慢的數據已存入: {OUTPUT_PATH}")
print("📊 附帶產出了一張 'Slow_Effect_Demo_3.0x.png'，你可以打開看看波形被拉長的物理效果！")