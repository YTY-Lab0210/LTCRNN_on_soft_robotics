import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

# 讓終端機保持乾淨
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

# ================= 1. 讀取 CSV 資料 =================
# ⚠️ 請將這裡替換成你實際的 CSV 檔名
csv_file_path = "Benchmark_Results_BPTT.csv" 
df = pd.read_csv(csv_file_path)

# 你的模型列表 (確保與 CSV 第一列的名稱完全一致)
model_configs = ['1D-CNN', 'SimpleRNN-8', 'LSTM-8', 'LTC-4']

# ================= 2. 轉換資料格式 (寬表格寫法) =================
data_to_plot = []
for name in model_configs:
    if name in df.columns:
        # 直接抓取該模型欄位底下的所有數值，並清除可能的空值 (NaN)
        model_data = df[name].dropna().values
        data_to_plot.append(model_data)
    else:
        print(f"⚠️ 警告：在 CSV 中找不到 '{name}'，請檢查名稱有無空白或大小寫差異！")
        data_to_plot.append(np.array([]))

# ================= 3. 繪製精美 Boxplot =================
print("\n📊 正在繪製多模型基準測試 Boxplot...")
fig, ax = plt.subplots(figsize=(10, 6), dpi=300) 

labels = model_configs

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
    if len(data_pts) == 0: continue 
    
    x_jitter = np.random.normal(i + 1, 0.04, size=len(data_pts))
    ax.scatter(x_jitter, data_pts, color='black', alpha=0.6, s=30, edgecolor='white', linewidth=0.8, zorder=2)

# 🌟 計算平均值，並動態把標籤放在該組數據的「最低分下方」
global_min = float('inf')
for i, data_pts in enumerate(data_to_plot):
    if len(data_pts) == 0: continue
        
    mean_val = np.mean(data_pts)        
    min_val = np.min(data_pts)          
    center_x = i + 1  
    
    if min_val < global_min:
        global_min = min_val
        
    # 將文字放在該組數據最低分的下方 (min_val - 2.0)
    ax.text(center_x, min_val - 2.0, f'Mean: {mean_val:.1f}%', 
             ha='center', va='top', fontsize=11, fontweight='bold', color='black',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#CCCCCC', boxstyle='round,pad=0.3'), zorder=4)

# 設定標題與軸標籤
ax.set_title("Model Architecture Performance Comparison (10 Runs) | BPTT", fontsize=15, fontweight='bold', pad=15)
ax.set_ylabel("Test Accuracy (%)", fontsize=14, fontweight='bold')

# 動態調整 Y 軸：確保最底下的標籤不會被切掉
ax.set_ylim([max(0, global_min - 10), 100])
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("Benchmark_Accuracy_Boxplot_BPTT.png", dpi=300, bbox_inches='tight')
print("📊 完美！基準測試 Boxplot 圖表已成功儲存為 Benchmark_Accuracy_Boxplot_BPTT.png")

plt.show()