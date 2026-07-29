import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ================= 1. 畫布設定 =================
fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300)
ax.axis('off')  # 關閉座標軸

# 顏色設定
color_vector = '#D6EAF8'  # 淺藍色 (1D Vector)
color_map = '#FADBD8'     # 淺粉橘色 (2D Feature Map)

# ================= 2. 繪圖輔助函數 =================
def draw_box(ax, x, y, width, height, color, title):
    """畫出代表張量 (Tensor) 的矩形方塊"""
    rect = patches.Rectangle((x, y), width, height, linewidth=1.5, 
                             edgecolor='#2C3E50', facecolor=color, alpha=0.9)
    ax.add_patch(rect)
    # 頂部張量維度標示 (拉高距離避免重疊)
    ax.text(x + width/2, y + height + 0.4, title, 
            ha='center', va='bottom', fontsize=14, fontweight='bold')

def draw_operation_with_arrow(ax, x_start, x_end, y_center, top_text, bottom_text):
    """畫出帶有箭頭的層級連接線，以及上下方的參數說明"""
    # 畫出帶有三角形箭頭的實線
    ax.annotate('', xy=(x_end, y_center), xytext=(x_start, y_center),
                arrowprops=dict(arrowstyle='-|>', lw=2.5, color='#7F8C8D', shrinkA=5, shrinkB=5))
    
    mid_x = (x_start + x_end) / 2
    # 上方說明文字 (加上白色半透明底框防遮擋)
    ax.text(mid_x, y_center + 0.5, top_text, 
            ha='center', va='bottom', fontsize=11, fontstyle='italic', color='#2C3E50',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))
    
    # 下方參數矩陣說明
    ax.text(mid_x, y_center - 0.5, bottom_text, 
            ha='center', va='top', fontsize=12, fontweight='bold', color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))

# ================= 3. 繪製各層架構 (加寬間距) =================

y_c = 4.0 # 設定所有箭頭與方塊的垂直中心基準線

# --- [Layer 0: Input] ---
draw_box(ax, x=0, y=1.5, width=1.0, height=5.0, color=color_vector, 
         title=r"$\mathbf{x} \in \mathbb{R}^{400 \times 5}$")

# 箭頭 1: Conv1D
draw_operation_with_arrow(ax, x_start=1.0, x_end=4.5, y_center=y_c, 
               top_text="Convolution 1D\nSize=5\nStride=1\nChannels=8",
               bottom_text=r"$\mathbf{W}_1 \in \mathbb{R}^{5 \times 5 \times 8}$" + "\n" + 
                           r"$\mathbf{b}_1 \in \mathbb{R}^8$" + "\n\n" + 
                           "Params: 208")

# --- [Layer 1: Conv1D Output] ---
draw_box(ax, x=4.5, y=1.8, width=2.5, height=4.4, color=color_map, 
         title=r"$\mathbf{H}_1 \in \mathbb{R}^{396 \times 8}$")

# 箭頭 2: Global Average Pooling
draw_operation_with_arrow(ax, x_start=7.0, x_end=10.5, y_center=y_c, 
               top_text="Global Average\nPooling 1D",
               bottom_text="Params: 0")

# --- [Layer 2: GAP Output] ---
draw_box(ax, x=10.5, y=3.2, width=1.0, height=1.6, color=color_vector, 
         title=r"$\mathbf{h}_{gap} \in \mathbb{R}^8$")

# 箭頭 3: Dense
draw_operation_with_arrow(ax, x_start=11.5, x_end=14.5, y_center=y_c, 
               top_text="Fully connected\nInput=8\nOutput=3\nSoftmax",
               bottom_text=r"$\mathbf{W}_2 \in \mathbb{R}^{8 \times 3}$" + "\n" + 
                           r"$\mathbf{b}_2 \in \mathbb{R}^3$" + "\n\n" + 
                           "Params: 27")

# --- [Layer 3: Final Output] ---
draw_box(ax, x=14.5, y=3.6, width=1.0, height=0.8, color=color_vector, 
         title=r"$\mathbf{f} \in \mathbb{R}^3$")

# ================= 4. 圖表總結與存檔 =================
total_params_text = "Total Trainable Params: 235\n(Extremely Lightweight Base)"
ax.text(15.5, 0.5, total_params_text, ha='right', va='bottom', 
        fontsize=13, fontweight='bold', color='#C0392B',
        bbox=dict(facecolor='white', edgecolor='#C0392B', boxstyle='round,pad=0.6'))

ax.set_xlim(-1, 16)
ax.set_ylim(0, 8.5)

plt.tight_layout()
plt.savefig("Lightweight_1DCNN_Fixed.png", dpi=300, bbox_inches='tight')
print("📊 完美修正！帶箭頭與排版對齊的圖表已儲存為 Lightweight_1DCNN_Fixed.png")

plt.show()