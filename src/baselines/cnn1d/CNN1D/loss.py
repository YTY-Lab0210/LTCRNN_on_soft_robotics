import matplotlib.pyplot as plt

# 建立與老師投影片一模一樣的 1x2 對比圖格式
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
fig.suptitle('Your Actual 1D-CNN (Stride=2) Training Results', fontsize=18, fontweight='bold', y=1.02)

# 顏色設定 (沿用原圖配色)
color_train = '#C46D4B'  # 橘褐
color_test = '#5FAAA0'   # 青綠

# ----------------- 左圖：Accuracy -----------------
# 投影片是畫 Error (1 - Accuracy)，我們直接畫習慣的 Accuracy
ax1.plot(history.history['accuracy'], color=color_train, linewidth=2.5)
ax1.plot(history.history['val_accuracy'], color=color_test, linewidth=2.5)

ax1.text(len(history.history['accuracy'])-2, history.history['accuracy'][-1]-0.05, 'Train', color=color_train, fontsize=12, fontweight='bold')
ax1.text(len(history.history['val_accuracy'])-2, history.history['val_accuracy'][-1]+0.02, 'Test', color=color_test, fontsize=12, fontweight='bold')

ax1.set_xlabel('Epochs', fontsize=14)
ax1.set_ylabel('Accuracy', fontsize=14)
ax1.set_title('a) Model Accuracy', fontsize=14, pad=10)
ax1.grid(True, linestyle='--', alpha=0.5)

# ----------------- 右圖：Loss (終結 U 型反彈) -----------------
ax2.plot(history.history['loss'], color=color_train, linewidth=2.5)
ax2.plot(history.history['val_loss'], color=color_test, linewidth=2.5)

ax2.text(len(history.history['loss'])-2, history.history['loss'][-1]+0.05, 'Train', color=color_train, fontsize=12, fontweight='bold')
ax2.text(len(history.history['val_loss'])-2, history.history['val_loss'][-1]+0.05, 'Test', color=color_test, fontsize=12, fontweight='bold')

ax2.set_xlabel('Epochs', fontsize=14)
ax2.set_ylabel('Loss', fontsize=14)
ax2.set_title('b) Model Loss (No Overfitting!)', fontsize=14, pad=10)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('your_real_1dcnn_curves.png', dpi=300, bbox_inches='tight')
plt.show()