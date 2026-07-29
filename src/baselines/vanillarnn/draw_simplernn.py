import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ================= 1. 畫布設定 =================
fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
ax.axis('off')

color_vector = '#D6EAF8'  # 淺藍色 (1D Vector)

# ================= 2. 繪圖輔助函數 =================
def draw_box(ax, x, y, width, height, color, title):
    rect = patches.Rectangle((x, y), width, height, linewidth=1.5, 
                             edgecolor='#2C3E50', facecolor=color, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height + 0.4, title, 
            ha='center', va='bottom', fontsize=14, fontweight='bold')

def draw_operation_with_arrow(ax, x_start, x_end, y_center, top_text, bottom_text):
    ax.annotate('', xy=(x_end, y_center), xytext=(x_start, y_center),
                arrowprops=dict(arrowstyle='-|>', lw=2.5, color='#7F8C8D', shrinkA=5, shrinkB=5))
    
    mid_x = (x_start + x_end) / 2
    ax.text(mid_x, y_center + 0.5, top_text, 
            ha='center', va='bottom', fontsize=11, fontstyle='italic', color='#2C3E50',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))
    
    ax.text(mid_x, y_center - 0.5, bottom_text, 
            ha='center', va='top', fontsize=12, fontweight='bold', color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))

# ================= 3. 繪製各層架構 =================
y_c = 4.0 

# --- [Layer 0: Input] ---
draw_box(ax, x=0, y=1.5, width=1.0, height=5.0, color=color_vector, 
         title=r"$\mathbf{x} \in \mathbb{R}^{400 \times 5}$")

# 箭頭 1: SimpleRNN (return_sequences=False)
# 參數計算: Input(5)*8 + Hidden(8)*8 + Bias(8) = 40 + 64 + 8 = 112
draw_operation_with_arrow(ax, x_start=1.0, x_end=5.5, y_center=y_c, 
               top_text="SimpleRNN\nUnits=8\n(return_sequences=False)",
               bottom_text=r"$\mathbf{W}_{in} \in \mathbb{R}^{5 \times 8}$" + "\n" + 
                           r"$\mathbf{W}_{rec} \in \mathbb{R}^{8 \times 8}$" + "\n" + 
                           r"$\mathbf{b} \in \mathbb{R}^8$" + "\n\n" + 
                           "Params: 112")

# --- [Layer 1: RNN Hidden State Output] ---
draw_box(ax, x=5.5, y=3.2, width=1.0, height=1.6, color=color_vector, 
         title=r"$\mathbf{h}_{rnn} \in \mathbb{R}^8$")

# 箭頭 2: Dense
# 參數計算: Input(8)*3 + Bias(3) = 24 + 3 = 27
draw_operation_with_arrow(ax, x_start=6.5, x_end=10.5, y_center=y_c, 
               top_text="Fully connected\nInput=8\nOutput=3\nSoftmax",
               bottom_text=r"$\mathbf{W}_{dense} \in \mathbb{R}^{8 \times 3}$" + "\n" + 
                           r"$\mathbf{b}_{dense} \in \mathbb{R}^3$" + "\n\n" + 
                           "Params: 27")

# --- [Layer 2: Final Output] ---
draw_box(ax, x=10.5, y=3.6, width=1.0, height=0.8, color=color_vector, 
         title=r"$\mathbf{f} \in \mathbb{R}^3$")

# ================= 4. 圖表總結與存檔 =================
total_params_text = "Total Trainable Params: 139\n(Traditional Baseline)"
ax.text(11.5, 0.5, total_params_text, ha='right', va='bottom', 
        fontsize=13, fontweight='bold', color='#2980B9',
        bbox=dict(facecolor='white', edgecolor='#2980B9', boxstyle='round,pad=0.6'))

ax.set_xlim(-1, 12)
ax.set_ylim(0, 8.5)

plt.tight_layout()
plt.savefig("Traditional_SimpleRNN_Architecture.png", dpi=300, bbox_inches='tight')
print("📊 SimpleRNN 架構圖已順利產出並儲存！")

plt.show()