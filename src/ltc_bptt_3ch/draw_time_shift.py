import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile

# 讓終端機繪圖暫存保持乾淨
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

# ================= 1. 讀取 CSV 資料 =================
csv_file = "TimeShift_Robustness_Raw_Data.csv"

if not os.path.exists(csv_file):
    print(f"⚠️ 找不到檔案 {csv_file}，請確認檔案路徑是否正確！")
else:
    print(f"🚀 成功讀取 {csv_file}，開始繪圖...")
    df_results = pd.read_csv(csv_file)

    # ================= 2. 繪圖設定 =================
    sns.set_theme(style="whitegrid") 
    fig, ax = plt.subplots(figsize=(15, 7), dpi=300)
    
    # 統一的配色盤
    model_palette = {
        '1D-CNN': '#ff7f0e', 
        'SimpleRNN-8': '#7f7f7f',
        'LSTM-8': '#d62728',
        'LTC-4': '#1f77b4'
    }

    # 確保 X 軸與圖例的順序
    x_order = ['Clean', '-1s', '+1s']
    hue_order = ['1D-CNN', 'SimpleRNN-8', 'LSTM-8', 'LTC-4']

    # 繪製分組箱型圖
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

    # ================= 3. 加入平均值標籤 =================
    offsets = [-0.2625, -0.0875, 0.0875, 0.2625]
    global_min = df_results['Accuracy (%)'].min()

    for x_idx, shift_label in enumerate(x_order):
        for hue_idx, model in enumerate(hue_order):
            subset = df_results[(df_results['Shift Level'] == shift_label) & (df_results['Model'] == model)]
            if not subset.empty:
                mean_val = subset['Accuracy (%)'].mean()
                min_val = subset['Accuracy (%)'].min()
                x_pos = x_idx + offsets[hue_idx]  
                
                ax.text(x_pos, min_val - 1.5, f'{mean_val:.1f}',  # 稍微往下移一點點，避免跟下鬚線黏太緊
                        ha='center', va='top', fontsize=9, fontweight='bold', color='black',
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=1),
                        zorder=10)

    # ================= 4. 外觀與輸出設定 =================
    # ax.set_title("OOD Robustness: Model Accuracy Under Temporal Shift", fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("Time Shift", fontsize=14, fontweight='bold')
    ax.set_ylabel("Test Accuracy (%)", fontsize=14, fontweight='bold')
    
    # 動態調整 Y 軸下界，留空間給平均值標籤
    ax.set_ylim([max(0, global_min - 12), 100])
    
    plt.legend(title='Model Architecture', title_fontsize='12', fontsize='11', loc='lower left')

    plt.tight_layout()
    save_name = "TimeShift_Robustness_Grouped_Boxplot.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"🎉 完美！時間平移測試圖表已成功儲存為 {save_name}")
    
    plt.show()