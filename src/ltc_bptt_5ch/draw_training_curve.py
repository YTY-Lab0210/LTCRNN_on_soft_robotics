import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tempfile

# 讓終端機繪圖暫存保持乾淨
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

# ================= 1. 設定檔案路徑與模型名稱 =================
# ⚠️ 請確保這裡的 CSV 檔名與你實際產生的檔名完全一致
history_files = {
    '1D-CNN': 'history_1dcnn.csv',
    'SimpleRNN-8': 'history_simplernn8.csv',
    'LSTM-8': 'history_lstm8.csv',
    'LTC-4': 'history_ltc4.csv'
}

# ================= 2. 統一的線條顏色設定 =================
# Training 用深藍色，Validation 用深紅色 (皆為實線，高對比易讀)
COLOR_TRAIN = '#1f77b4'  # Deep Blue
COLOR_VAL = '#d62728'    # Deep Red
LINE_WIDTH = 2.5         # 加粗線條讓它更好看清楚

# ================= 3. 開始繪圖 =================
print("📊 正在讀取 CSV 並繪製各模型的 Training Dynamics 趨勢圖...")

for model_name, file_path in history_files.items():
    if not os.path.exists(file_path):
        print(f"⚠️ 找不到 {model_name} 的訓練紀錄 {file_path}，跳過繪製。")
        continue
        
    df = pd.read_csv(file_path)
    
    # 🔑 幫每個模型開一張新的畫布 (1x2 雙拼)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # 左圖：Loss (皆為實線 linestyle='-')
    axes[0].plot(df['epoch'], df['loss'], color=COLOR_TRAIN, linestyle='-', linewidth=LINE_WIDTH, alpha=0.9)
    axes[0].plot(df['epoch'], df['val_loss'], color=COLOR_VAL, linestyle='-', linewidth=LINE_WIDTH, alpha=0.9)
    
    # 右圖：Accuracy (自動轉為百分比)
    train_acc = df['accuracy'] * 100 if df['accuracy'].max() <= 1.0 else df['accuracy']
    val_acc = df['val_accuracy'] * 100 if df['val_accuracy'].max() <= 1.0 else df['val_accuracy']
    
    # (皆為實線 linestyle='-')
    axes[1].plot(df['epoch'], train_acc, color=COLOR_TRAIN, linestyle='-', linewidth=LINE_WIDTH, alpha=0.9)
    axes[1].plot(df['epoch'], val_acc, color=COLOR_VAL, linestyle='-', linewidth=LINE_WIDTH, alpha=0.9)

    # 統一的圖例外觀
    legend_elements = [
        Line2D([0], [0], color=COLOR_TRAIN, lw=LINE_WIDTH, linestyle='-', label='Training'),
        Line2D([0], [0], color=COLOR_VAL, lw=LINE_WIDTH, linestyle='-', label='Validation')
    ]

    # 左圖 (Loss) 外觀設定
    axes[0].set_title(f"{model_name} - Loss", fontsize=16, fontweight='bold', pad=15)
    axes[0].set_xlabel("Epoch", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Loss (Categorical Crossentropy)", fontsize=14, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)

    # 右圖 (Accuracy) 外觀設定
    axes[1].set_title(f"{model_name} - Accuracy", fontsize=16, fontweight='bold', pad=15)
    axes[1].set_xlabel("Epoch", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Accuracy (%)", fontsize=14, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_ylim([0, 102]) 
    axes[1].legend(handles=legend_elements, loc='lower right', fontsize=12, framealpha=0.9)

    # 儲存獨立圖檔
    plt.tight_layout()
    save_name = f"Training_Dynamics_{model_name}.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"🎉 {model_name} 趨勢圖已成功儲存為 {save_name}")

# 一次顯示剛畫好的圖 (如果你在 Jupyter 或終端機執行)
plt.show()