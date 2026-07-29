import matplotlib.pyplot as plt
import numpy as np

# ================= 1. 資料準備 =================
# 橫軸：訓練集數量 (視為等距類別)
data_nums = [1, 3, 5, 10, 30, 50]
x_pos = np.arange(len(data_nums))

# 縱軸：測試準確率 (%)
# Mutation rate = 0.20
mu_020_LTC1 = [43.33, 66.67, 73.33, 63.33, 70.00, 66.67]
mu_020_LTC2 = [46.67, 76.67, 80.00, 70.00, 73.33, 73.33]
mu_020_LTC3 = [56.67, 70.00, 73.33, 63.33, 73.33, 86.67]

# Mutation rate = 0.15
mu_015_LTC1 = [40.00, 60.00, 83.33, 73.33, 83.33, 70.00]
mu_015_LTC2 = [43.33, 70.00, 66.67, 76.67, 86.67, 83.33]
mu_015_LTC3 = [36.67, 80.00, 90.00, 83.33, 76.67, 66.67]

# Mutation rate = 0.10
mu_010_LTC1 = [50.00, 66.67, 80.00, 60.00, 70.00, 80.00]
mu_010_LTC2 = [43.33, 56.67, 90.00, 76.67, 90.00, 86.67]
mu_010_LTC3 = [40.00, 53.33, 76.67, 90.00, 86.67, 90.00]

# ================= 2. 畫布與樣式設定 =================
# 建立 1x3 的子圖，尺寸設定為適合寬版放入論文的比例
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

# 共用的繪圖參數
plot_kwargs = {
    'LTC1': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '--', 'linewidth': 2, 'markersize': 8, 'label': '1-Neuron (LTC1)'},
    'LTC2': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-', 'linewidth': 2.5, 'markersize': 8, 'label': '2-Neuron (LTC2)'},
    'LTC3': {'color': '#2ca02c', 'marker': '^', 'linestyle': ':', 'linewidth': 2, 'markersize': 9, 'label': '3-Neuron (LTC3)'}
}

# 🌟 修正警告：在字串最前面加上 r，變成 Raw String
datasets = [
    (axes[0], r"Mutation Rate ($\mu$) = 0.20", mu_020_LTC1, mu_020_LTC2, mu_020_LTC3),
    (axes[1], r"Mutation Rate ($\mu$) = 0.15", mu_015_LTC1, mu_015_LTC2, mu_015_LTC3),
    (axes[2], r"Mutation Rate ($\mu$) = 0.10", mu_010_LTC1, mu_010_LTC2, mu_010_LTC3)
]

# ================= 3. 執行繪圖 =================
for ax, title, ltc1, ltc2, ltc3 in datasets:
    # 畫出三條線
    ax.plot(x_pos, ltc1, **plot_kwargs['LTC1'])
    ax.plot(x_pos, ltc2, **plot_kwargs['LTC2'])
    ax.plot(x_pos, ltc3, **plot_kwargs['LTC3'])
    
    # 標題與軸標籤設定
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Number of Training Samples", fontsize=12, fontweight='bold')
    
    # 設定 X 軸的刻度為等距的類別標籤
    ax.set_xticks(x_pos)
    ax.set_xticklabels(data_nums, fontsize=11)
    
    # 設定網格與 Y 軸範圍
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(30, 95)
    
    # 特別標示出 90% 的黃金基準線 (選用)
    ax.axhline(y=90.0, color='red', linestyle='-', linewidth=1, alpha=0.3)
    
    # 🌟 修復裁切問題：將圖例獨立放進每個子圖的右下角
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9, edgecolor='gray')

# 只有最左邊的圖需要 Y 軸標籤
axes[0].set_ylabel("Test Accuracy (%)", fontsize=12, fontweight='bold')

# ================= 4. 顯示與存檔 =================
plt.suptitle("Ablation Study: Impact of Model Capacity, Training Size, and Mutation Rate", 
             fontsize=16, fontweight='bold', y=1.05)

# 使用更安全的邊界設定取代單純的 tight_layout
plt.subplots_adjust(top=0.88, bottom=0.15, wspace=0.1)
plt.savefig("Ablation_Study_Accuracy.png", dpi=300, bbox_inches='tight')
print("✅ 圖表已成功存檔為 Ablation_Study_Accuracy.png")

plt.show()