import matplotlib.pyplot as plt
import numpy as np

# ================= 1. 實驗數據 (平均值 Mean 與 標準差 Std) =================
x_labels = ['60', '30', '15', '6', '3']
x_pos = np.arange(len(x_labels))

# (以下為模擬示範數據，請替換為你真實的 np.mean 和 np.std 結果)
# LSTM-8：參數多，資料少時不僅平均值暴跌，標準差(變異)也會暴增
acc_lstm8_mean = np.array([94.5, 88.2, 75.1, 52.3, 38.4])
acc_lstm8_std  = np.array([ 1.2,  2.5,  5.8, 12.4, 18.5]) 

# 1D-CNN：同樣面臨過擬合與不穩定
acc_cnn8_mean  = np.array([93.1, 89.5, 80.2, 60.1, 45.2])
acc_cnn8_std   = np.array([ 1.0,  2.1,  4.5,  9.8, 15.2])

# LTC-4：參數極少，ODE約束，平均值抗跌且標準差(變異)極小
acc_ltc4_mean  = np.array([92.7, 91.5, 89.0, 83.5, 76.2])
acc_ltc4_std   = np.array([ 0.8,  1.1,  1.5,  2.8,  4.5]) 

# ================= 2. 繪製帶有陰影的折線圖 =================
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

def plot_with_shadow(ax, x, mean, std, color, marker, linestyle, label):
    # 畫出平均值的實線
    ax.plot(x, mean, marker=marker, markersize=8, linewidth=2.5, linestyle=linestyle, color=color, label=label)
    # 畫出標準差的透明陰影 (透明度 alpha=0.15)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, edgecolor='none')

# 依序畫上三條線與陰影 (為了畫面乾淨，這裡示範三種最具代表性的模型)
plot_with_shadow(ax, x_pos, acc_lstm8_mean, acc_lstm8_std, '#d62728', 's', '--', 'LSTM-8 (475 params)')
plot_with_shadow(ax, x_pos, acc_cnn8_mean,  acc_cnn8_std,  '#ff7f0e', '^', '--', '1D-CNN (235 params)')
plot_with_shadow(ax, x_pos, acc_ltc4_mean,  acc_ltc4_std,  '#2ca02c', 'o', '-',  'LTC-4 (75 params)')

# ================= 3. 圖表美化 =================
ax.set_title("Data Efficiency and Robustness (Mean ± Std over 10 Runs)", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Number of Training Samples per Class", fontsize=14, fontweight='bold')
ax.set_ylabel("Test Accuracy (%)", fontsize=14, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=11, loc='lower left', framealpha=0.9)

plt.tight_layout()
plt.savefig("Few_Shot_Efficiency_With_Variance.png", dpi=300, bbox_inches='tight')
print("📊 完美！帶有標準差陰影的折線圖已儲存為 Few_Shot_Efficiency_With_Variance.png")

plt.show()