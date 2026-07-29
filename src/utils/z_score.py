import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ================= 全局設定 =================
TARGET_LINES = 400

# 🌟 1. 設定你的「根目錄」
ROOT_DIR = r'C:\Users\HAO\Desktop\YTY_from_macbook\dataset_xx2020_new_new_new'

# 🌟 2. 設定你要處理的資料集數量清單
DATA_NUMS = [60] 

# ================= 1. 讀取與檔名追蹤函式 =================
def load_sensor_data_with_names(folder_path):
    """讀取資料並同時回傳檔名，方便後續對應存檔"""
    all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    signal_list, filename_list = [], []
    
    if len(all_files) == 0:
        print(f"⚠️ 警告：在 {folder_path} 中找不到任何 CSV 檔案！")
        return np.array([]), []

    for file in all_files:
        filename = os.path.basename(file)
        df = pd.read_csv(file)
        features = df[['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']].values

        if features.shape == (TARGET_LINES, 5):
            signal_list.append(features)
            filename_list.append(filename) 
        else:
            print(f"🚨 警告：已剔除檔案 {filename}！(資料維度為 {features.shape}，不符合預期的 ({TARGET_LINES}, 5))")
            
    return np.array(signal_list), filename_list

# ================= 2. 論文級別的波形視覺化 =================
def plot_zscore_waveforms(train_scaled, val_scaled, test_scaled, train_names, val_names, test_names, target_base_path, data_num):
    """抽取各資料集的第一筆資料，繪製 Z-score 標準化後的波形圖"""
    # 如果資料是空的，就不畫圖
    if len(train_scaled) == 0 or len(val_scaled) == 0 or len(test_scaled) == 0:
        print("⚠️ 資料不足以繪圖，跳過此步驟。")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    datasets = [
        ('Training', train_scaled[0], train_names[0]),
        ('Validation', val_scaled[0], val_names[0]),
        ('Test', test_scaled[0], test_names[0])
    ]
    
    for i, (split_name, data, filename) in enumerate(datasets):
        ax = axes[i]
        for finger_idx in range(5):
            ax.plot(data[:, finger_idx], label=finger_names[finger_idx], color=colors[finger_idx], linewidth=1.5)
            
        ax.set_title(f"{split_name} Set Sample: {filename} (Z-Score Normalized)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time Steps (0-400)", fontsize=10)
        ax.set_ylabel("Amplitude (Z-Score)", fontsize=10)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', ncol=5, fontsize=9)
        ax.set_ylim(-4, 4) 
        
    # 調整排版，避免重疊
    plt.subplots_adjust(hspace=0.4)
    
    # 🌟 針對每個資料集存下專屬的檔名
    save_path = os.path.join(target_base_path, f'Z_score_Waveforms_DataNum_{data_num}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 波形比對圖已儲存為：{save_path}")
    
    # 在批次處理時，建議把 plt.show() 關掉，不然要手動關閉視窗才會跑下一個
    plt.close()

# ================= 3. 主處理與存檔函式 =================
def process_and_save_data(source_base_path, target_base_path, data_num):
    print(f"\n{'='*50}")
    print(f"🚀 開始處理資料集：Data Num = {data_num}")
    print(f"📂 來源：{source_base_path}")
    print(f"📂 目標：{target_base_path}")
    print(f"{'='*50}")
    
    train_raw, train_names = load_sensor_data_with_names(os.path.join(source_base_path, 'training'))
    val_raw, val_names     = load_sensor_data_with_names(os.path.join(source_base_path, 'validation'))
    test_raw, test_names   = load_sensor_data_with_names(os.path.join(source_base_path, 'test'))

    # 檢查是否有讀到資料
    if len(train_raw) == 0:
        print(f"❌ 錯誤：無法從 {source_base_path} 讀取 Training 資料，跳過此資料集。")
        return

    print("🧮 進行 Z-score 標準化計算 (Scaler fitted on Training Set)...")
    num_features = 5
    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train_raw.reshape(-1, num_features)).reshape(-1, TARGET_LINES, num_features)
    val_scaled   = scaler.transform(val_raw.reshape(-1, num_features)).reshape(-1, TARGET_LINES, num_features)
    test_scaled  = scaler.transform(test_raw.reshape(-1, num_features)).reshape(-1, TARGET_LINES, num_features)

    def save_to_csv(scaled_data, filenames, split_name):
        target_folder = os.path.join(target_base_path, split_name)
        os.makedirs(target_folder, exist_ok=True) 
        columns = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        
        for i, data_matrix in enumerate(scaled_data):
            df_saved = pd.DataFrame(data_matrix, columns=columns)
            save_path = os.path.join(target_folder, filenames[i])
            df_saved.to_csv(save_path, index=False)
            
        print(f"✅ {split_name:<10} 已存檔完成 ({len(filenames)} 筆)")

    print("💾 開始匯出標準化數據...")
    save_to_csv(train_scaled, train_names, 'training')
    save_to_csv(val_scaled, val_names, 'validation')
    save_to_csv(test_scaled, test_names, 'test')
    
    print("🎨 正在繪製波形圖抽樣比對...")
    plot_zscore_waveforms(train_scaled, val_scaled, test_scaled, train_names, val_names, test_names, target_base_path, data_num)
    print(f"🎉 資料集 {data_num} 處理完畢！\n")


# ================= 4. 執行批次處理迴圈 =================
if __name__ == "__main__":
    for num in DATA_NUMS:
        # 動態組合來源與目標資料夾的路徑
        source_path = os.path.join(ROOT_DIR, f'dataset_{num}2020')
        target_path = os.path.join(ROOT_DIR, f'dataset_{num}2020_zscore')
        
        # 檢查來源資料夾是否存在
        if not os.path.exists(source_path):
            print(f"⚠️ 找不到來源資料夾：{source_path}，已跳過。")
            continue
            
        process_and_save_data(source_path, target_path, num)
        
    print("\n🏁 所有資料集皆已 Z-score 處理完畢！準備進入演算法激戰階段！")