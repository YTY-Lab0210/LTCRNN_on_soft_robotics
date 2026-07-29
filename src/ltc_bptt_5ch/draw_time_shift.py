import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile

# 讓終端機保持乾淨
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

# ================= 1. 讀取 CSV 資料 =================
# ⚠️ 請確認檔名與你輸出的 CSV 檔名一致
csv_file_path = "TimeShift_Robustness_Raw_Data.csv"
df_results = pd.read_csv(csv_file_path)

# ================= 2. 設定繪圖參數與色系 =================
sns.set_theme(style="whitegrid") 
fig, ax = plt.subplots(figsize=(15, 7), dpi=300)

# 你的模型顏色設定 (字典格式，Seaborn 會自動安全對應)
model_palette = {
    '1D-CNN': '#ff7f0e', 
    'SimpleRNN-8': '#7f7f7f',
    'LSTM-8': '#d62728',
    'LTC-4': '#1f77b4'
}

# ⚠️ 這裡的 x_order 必須和 CSV 裡的 'Shift Level' 字串完全一樣
# 根據你的圖表，這裡預設為 Frames。如果你存的是 '-1s'，請改成 ['Clean', '-1s', '+1s']
x_order = ['Clean', '-1s', '+1s']
hue_order = ['1D-CNN', 'SimpleRNN-8', 'LSTM-8', 'LTC-4']

# ================= 3. 繪製分組箱型圖 =================
sns.boxplot(
    data=df_results, 
    x='Shift Level', 
    y='Accuracy (%)', 
    hue='Model', 
    order=x_order,
    hue_order=hue_order,
    ax=ax, 
    palette=model_palette,
    linewidth=1.5,
    width=0.7,
    fliersize=4
)

# ================= 4. 加入平均值標籤 =================
# 用來微調 4 個模型在 X 軸上的文字位置
offsets = [-0.2625, -0.0875, 0.0875, 0.2625]
global_min = df_results['Accuracy (%)'].min()

for x_idx, shift_label in enumerate(x_order):
    for hue_idx, model in enumerate(hue_order):
        # 篩選特定位移與特定模型的數據
        subset = df_results[(df_results['Shift Level'] == shift_label) & (df_results['Model'] == model)]
        if not subset.empty:
            mean_val = subset['Accuracy (%)'].mean()
            min_val = subset['Accuracy (%)'].min()
            x_pos = x_idx + offsets[hue_idx]  
            
            # 將平均值文字放在該箱子的最低分下方 (微調 -2.0 避免擋住鬍鬚)
            ax.text(x_pos, min_val - 2.0, f'{mean_val:.1f}', 
                    ha='center', va='top', fontsize=9, fontweight='bold', color='black',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=1),
                    zorder=10)

# ================= 5. 設定標題與軸標籤 =================
ax.set_title("OOD Robustness: Model Accuracy Under Temporal Shift", fontsize=18, fontweight='bold', pad=15)
ax.set_xlabel("Time Shift (Frames)", fontsize=14, fontweight='bold')
ax.set_ylabel("Test Accuracy (%)", fontsize=14, fontweight='bold')

# 動態調整 Y 軸，避免最下方的標籤被切掉，上限設為 102 留出視覺空間
ax.set_ylim([max(0, global_min - 12), 102])

plt.legend(title='Model Architecture', title_fontsize='12', fontsize='11', loc='lower left')

plt.tight_layout()
plt.savefig("TimeShift_Robustness_Grouped_Boxplot.png", dpi=300, bbox_inches='tight')
print("📊 完美！時間平移測試圖表已成功儲存為 TimeShift_Robustness_Grouped_Boxplot.png")

plt.show()