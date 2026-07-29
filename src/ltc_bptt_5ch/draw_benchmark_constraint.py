import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 自動尋找並讀取 CSV 檔案 =================
csv_filename = "Benchmark_Results_BPTT(in).csv"

# 防呆：如果還沒改名，就去抓原本的預設檔名
if not os.path.exists(csv_filename):
    csv_filename = "Benchmark_Results_BPTT.csv"

if not os.path.exists(csv_filename):
    raise FileNotFoundError(f"❌ 錯誤：在當前目錄下找不到任何相關的 CSV 檔案！請確認檔案是否存在。")

print(f"📬 成功找到數據源！正在從 {csv_filename} 讀取 10 Runs 實驗結果...")
df = pd.DataFrame()
df_raw = pd.read_csv(csv_filename)

# ================= 2. 設定模型對照與專屬色系 =================
# 確保按照這 4 個模型的順序排列
model_configs = ['1D-CNN', 'SimpleRNN-5', 'LSTM-2', 'LTC-4']
colors = ['#9467bd', '#8c564b', '#e377c2', '#1f77b4'] # 紫、棕、粉紅、藍

# 萃取對應模型的數據
data_to_plot = [df_raw[name].dropna().values for name in model_configs]

# ================= 3. 建立論文級畫布 =================
fig, ax = plt.subplots(figsize=(14, 4), dpi=300) # 完美復刻 14:4 比例

# 繪製基礎箱型圖
bplot = ax.boxplot(data_to_plot, patch_artist=True, labels=model_configs, 
                   boxprops=dict(facecolor='white', color='black', linewidth=1.5),
                   medianprops=dict(color='red', linewidth=2), # 鮮紅中位數線
                   whiskerprops=dict(linewidth=1.5),
                   capprops=dict(linewidth=1.5),
                   flierprops=dict(marker='o', color='black', alpha=0.5))

# 幫各個模型的箱子塗上相對應的精緻色彩
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6) # 微透明度提升質感

# ================= 4. 動態計算位置並標註 Mean 平均值 =================
y_min = ax.get_ylim()[0] 
y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
text_y_pos = y_min + (y_range * 0.02) # 固定在圖表底部上方 2% 的高度

for i, name in enumerate(model_configs):
    avg_acc = np.mean(df_raw[name])
    ax.text(i + 1, text_y_pos, f'Mean: {avg_acc:.2f}%', 
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.9))

# ================= 5. 圖表細節美化與存檔 =================
ax.set_title("Model Architecture Performance Comparison (10 Runs) | BPTT", fontsize=15, fontweight='bold', pad=15)
ax.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.7) # 僅開啟 Y 軸虛線橫規

# 稍微拉寬 Y 軸下限，確保最下方的 Mean 標籤不會與圖表外框相撞
ax.set_ylim(y_min - (y_range * 0.05), ax.get_ylim()[1])

plt.tight_layout()

# 統一儲存為受限版本圖片名稱
output_img_name = "Benchmark_Accuracy_Boxplot_BPTT_under_constraint.png"
plt.savefig(output_img_name, dpi=300, bbox_inches='tight')
print(f"🎨 完美！圖表已根據 CSV 真實數據重新繪製完畢，並成功儲存為：{output_img_name}")

plt.show()