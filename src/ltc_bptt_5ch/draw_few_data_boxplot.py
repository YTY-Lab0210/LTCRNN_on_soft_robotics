import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 讀取你剛剛辛苦跑完的 CSV 數據
df_results = pd.DataFrame(pd.read_csv("Few_Shot_Raw_Data_v1.csv"))

# 2. 準備畫布與顏色
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
custom_palette = {
    '1D-CNN': '#ff7f0e',
    'SimpleRNN-8': '#7f7f7f',
    'LSTM-8': '#d62728',
    'LTC-4': '#2ca02c',
    'LTC-8': '#1f77b4'
}

# 3. 畫圖 (這次 X 軸會把 60 到 3 全畫出來)
sns.boxplot(
    data=df_results, 
    x='Samples_Per_Class', 
    y='Accuracy', 
    hue='Model', 
    palette=custom_palette, 
    order=[60, 30, 15, 6, 3], # 🌟 完整顯示所有數量
    hue_order=['1D-CNN', 'SimpleRNN-8', 'LSTM-8', 'LTC-4', 'LTC-8'],
    ax=ax,
    width=0.6, 
    fliersize=5, 
    linewidth=1.2
)

# 4. 圖表美化
ax.set_title("Data Efficiency and Few-Shot Learning Distribution (10 Independent Runs)", 
             fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Number of Training Samples per Class", fontsize=14, fontweight='bold')
ax.set_ylabel("Test Accuracy (%)", fontsize=14, fontweight='bold')
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
ax.tick_params(axis='both', labelsize=12)

min_acc = df_results['Accuracy'].min()
ax.set_ylim(max(0, min_acc - 5), 100)

plt.legend(title='Neural Architecture', title_fontsize='12', fontsize='11', 
           loc='lower left', framealpha=0.9)

plt.tight_layout()
plt.savefig("Few_Shot_Efficiency_Boxplot_Fixed.png", dpi=300, bbox_inches='tight')
print("📊 完美！修復版的箱型圖已儲存為 Few_Shot_Efficiency_Boxplot_Fixed.png")
plt.show()