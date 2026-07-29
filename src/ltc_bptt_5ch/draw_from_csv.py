import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 讀取 CSV
df_results = pd.read_csv("Few_Shot_Raw_Data.csv")

# 2. 自動偵測模型名稱並設定顏色
unique_models = df_results['Model'].unique()
print("DataFrame 實際包含的模型名稱:", unique_models)

# 定義顏色對應表 (確保包含所有可能出現的名稱)
base_palette = {
    '1D-CNN': '#ff7f0e',
    'SimpleRNN-8': '#7f7f7f',
    'LSTM-8': '#d62728',
    'LTC-4': '#1f77b4'
}

# 篩選出字典裡有的顏色
current_palette = {model: base_palette.get(model, '#000000') for model in unique_models}

# 3. 繪圖
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

sns.boxplot(
    data=df_results, 
    x='Samples_Per_Class', 
    y='Accuracy', 
    hue='Model', 
    palette=current_palette, # 使用動態產生的 palette
    order=[60, 30, 15, 6, 3], 
    ax=ax,
    width=0.6,
    fliersize=5,
    linewidth=1.2
)

# ... 後面美化代碼保持不變 ...
ax.set_title("Data Efficiency and Few-Shot Learning Distribution", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Number of Training Samples per Class", fontsize=14, fontweight='bold')
ax.set_ylabel("Test Accuracy (%)", fontsize=14, fontweight='bold')

plt.legend(title='Neural Architecture', title_fontsize='12', fontsize='11', loc='lower left')
plt.tight_layout()
plt.savefig("Few_Shot_Efficiency_Boxplot.png", dpi=300)
plt.show()