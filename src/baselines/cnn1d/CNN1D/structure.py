import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_mnist_1d_style_network():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(-1, 8)
    ax.axis('off')

    # 顏色設定 (致敬原圖風格)
    color_io = '#D1E8E2'      # 輸入輸出: 淡青色
    color_hidden = '#D9A566'  # 隱藏層: 紅土磚色
    color_edge = '#2C3E50'    # 邊框顏色

    def draw_dot_grid(ax, x_start, y_start, cols, rows, cell_size=0.08, color='#D9A566'):
        """在圖層方塊內部繪製圓點矩陣效果"""
        for r in range(rows):
            for c in range(cols):
                cx = x_start + c * cell_size + cell_size/2
                cy = y_start + r * cell_size + cell_size/2
                circle = patches.Circle((cx, cy), radius=cell_size*0.3,
                                       facecolor='white', edgecolor=color, linewidth=0.5, alpha=0.9)
                ax.add_patch(circle)

    # 1. Input Layer: x (400 x 5) -> 簡化繪製為一長條資料
    ax.add_patch(patches.Rectangle((0.8, 0.5), 0.2, 6.5, facecolor=color_io, edgecolor=color_edge, linewidth=1.5, zorder=2))
    draw_dot_grid(ax, 0.8, 0.5, 2, 50, cell_size=0.12, color=color_io) # 繪製示意點
    ax.text(0.6, 7.2, r'$\mathbf{x} \in \mathbb{R}^{400 \times 5}$', fontsize=14, fontweight='bold')
    ax.text(1.5, -0.4, r'$\mathbf{\Omega}_0 \in \mathbb{R}^{1 \times 3 \times 5 \times 15}$' + '\n' + r'$\mathbf{\beta}_0 \in \mathbb{R}^{15}$', fontsize=11, ha='center')

    # 2. Conv Layer 1: H1 (199 x 15)
    ax.add_patch(patches.Rectangle((2.0, 1.0), 1.8, 4.5, facecolor=color_hidden, edgecolor=color_edge, linewidth=1.5, zorder=2,alpha=0.5))
    draw_dot_grid(ax, 2.0, 1.0, 15, 37, cell_size=0.12, color=color_hidden)
    ax.text(3.0, 5.6, r'$\mathbf{H}_1 \in \mathbb{R}^{199 \times 15}$', fontsize=14, fontweight='bold', ha='center')
    ax.text(4.4, -0.0, r'$\mathbf{\Omega}_1 \in \mathbb{R}^{15 \times 3 \times 15}$' + '\n' + r'$\mathbf{\beta}_1 \in \mathbb{R}^{15}$', fontsize=11, ha='center')

    # 註釋第一層卷積參數
    ax.text(1.50, 3.2, 'Convolution\nSize=3\nStride=2\nChannels=15', fontsize=9, ha='center', fontstyle='italic', color='#555')

    # 3. Conv Layer 2: H2 (99 x 15)
    ax.add_patch(patches.Rectangle((4.8, 1.8), 1.8, 2.8, facecolor=color_hidden, edgecolor=color_edge, linewidth=1.5, zorder=2,alpha=0.5))
    draw_dot_grid(ax, 4.8, 1.8, 15, 23, cell_size=0.12, color=color_hidden)
    ax.text(5.7, 4.8, r'$\mathbf{H}_2 \in \mathbb{R}^{99 \times 15}$', fontsize=14, fontweight='bold', ha='center')
    ax.text(7.2, 0.6, r'$\mathbf{\Omega}_2 \in \mathbb{R}^{15 \times 3 \times 15}$' + '\n' + r'$\mathbf{\beta}_2 \in \mathbb{R}^{15}$', fontsize=11, ha='center')

    # 註釋第二層卷積參數
    ax.text(4.3, 3.2, 'Convolution\nSize=3\nStride=2\nChannels=15', fontsize=9, ha='center', fontstyle='italic', color='#555')

    # 4. Conv Layer 3: H3 (49 x 15)
    ax.add_patch(patches.Rectangle((7.6, 2.3), 1.6, 1.6, facecolor=color_hidden, edgecolor=color_edge, linewidth=1.5, zorder=2, alpha=0.5))
    draw_dot_grid(ax, 7.6, 2.3, 13, 13, cell_size=0.12, color=color_hidden)
    ax.text(8.4, 4.1, r'$\mathbf{H}_3 \in \mathbb{R}^{49 \times 15}$', fontsize=14, fontweight='bold', ha='center')
    ax.text(9.8, 1.0, r'$\mathbf{\Omega}_3 \in \mathbb{R}^{3 \times 735}$' + '\n' + r'$\mathbf{\beta}_3 \in \mathbb{R}^{3}$', fontsize=11, ha='center')

    # 註釋第三層卷積參數
    ax.text(7.1, 3.0, 'Convolution\nSize=3\nStride=2\nChannels=15', fontsize=9, ha='center', fontstyle='italic', color='#555')

    # 5. Output Layer: f (3 classes)
    ax.add_patch(patches.Rectangle((10.2, 2.3), 0.2, 1.6, facecolor=color_io, edgecolor=color_edge, linewidth=1.5, zorder=2))
    draw_dot_grid(ax, 10.2, 2.3, 1, 10, cell_size=0.15, color=color_io)
    ax.text(10.3, 4.1, r'$\mathbf{f} \in \mathbb{R}^{3}$', fontsize=14, fontweight='bold', ha='center')

    ax.text(9.7, 2.8, 'Fully connected\nInput=735\nOutput=3\nSoftmax', fontsize=9, ha='center', fontstyle='italic', color='#555')

    # 6. 頂部大箭頭 (代表資料流向)
    ax.annotate('', xy=(10.2, 6.0), xytext=(1.2, 6.0),
                arrowprops=dict(arrowstyle="->", color='#BDC3C7', lw=4))

    plt.title("Tactile Glove 1D-CNN Architecture (Adapted from MNIST-1D)", fontsize=18, fontweight='bold', pad=20)

    # 自動儲存高解析度圖片
    plt.savefig('tactile_glove_network_architecture.png', dpi=300, bbox_inches='tight')
    # plt.show()

# 執行繪圖
draw_mnist_1d_style_network()