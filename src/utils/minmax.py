import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= 全局設定 =================
TARGET_LINES = 400

# 🌟 1. 設定你的「根目錄」
ROOT_DIR = r'C:\Users\HAO\Desktop\YTY_from_macbook\dataset_xx2020'

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
            
    # 回傳的 numpy array 形狀會是 (樣本數, 400, 5)
    return np.array(signal_list), filename_list

# ================= 2. 論文級別的波形視覺化 (相對零點版) =================
def plot_zeroed_waveforms(train_scaled, val_scaled, test_scaled, train_names, val_names, test_names, target_base_path, data_num):
    """抽取各資料集的第一筆資料，繪製扣除第一列後的波形圖"""
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
            
        ax.set_title(f"{split_name} Set Sample: {filename} (First Row Subtracted)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time Steps (0-400)", fontsize=10)
        ax.set_ylabel("Amplitude (Relative to Step 0)", fontsize=10)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.8) # 強化 0 的基準線
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', ncol=5, fontsize=9)
        # 移除了硬寫死 Y 軸範圍的設定，讓 matplotlib 根據你的原始 ADC 數值自動縮放
        
    plt.subplots_adjust(hspace=0.4)
    
    save_path = os.path.join(target_base_path, f'Zeroed_Waveforms_DataNum_{data_num}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 波形比對圖已儲存為：{save_path}")
    
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

    if len(train_raw) == 0:
        print(f"❌ 錯誤：無法從 {source_base_path} 讀取 Training 資料，跳過此資料集。")
        return

    print("🧮 進行相對零點校正 (每筆資料扣除自身的第一列)...")

    def process_data(data_3d):
        # 1. 相對零點校正 (扣除第一列)
        first_step_values = data_3d[:, 0:1, :]
        zeroed = data_3d - first_step_values
        
        # 2. 🌟 關鍵：死區 (Deadzone) 處理
        # 絕對值小於 2 的，全部設為 0；絕對值大於 2 的，才保留數值
        # 這樣可以徹底過濾掉微小的傳感器飄移與雜訊
        processed = zeroed / 100
        return processed

    train_zeroed = process_data(train_raw)
    val_zeroed   = process_data(val_raw)
    test_zeroed  = process_data(test_raw)

    def save_to_csv(processed_data, filenames, split_name):
        target_folder = os.path.join(target_base_path, split_name)
        os.makedirs(target_folder, exist_ok=True) 
        columns = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        
        for i, data_matrix in enumerate(processed_data):
            df_saved = pd.DataFrame(data_matrix, columns=columns)
            save_path = os.path.join(target_folder, filenames[i])
            df_saved.to_csv(save_path, index=False)
            
        print(f"✅ {split_name:<10} 已存檔完成 ({len(filenames)} 筆)")

    print("💾 開始匯出歸零後的數據...")
    save_to_csv(train_zeroed, train_names, 'training')
    save_to_csv(val_zeroed, val_names, 'validation')
    save_to_csv(test_zeroed, test_names, 'test')
    
    print("🎨 正在繪製波形圖抽樣比對...")
    plot_zeroed_waveforms(train_zeroed, val_zeroed, test_zeroed, train_names, val_names, test_names, target_base_path, data_num)
    print(f"🎉 資料集 {data_num} 處理完畢！\n")

# ================= 4. 執行批次處理迴圈 =================
if __name__ == "__main__":
    for num in DATA_NUMS:
        source_path = os.path.join(ROOT_DIR, f'dataset_{num}2020')
        
        # 🌟 將目標資料夾命名為 _zeroed，代表這是以第一列為基準歸零的資料
        target_path = os.path.join(ROOT_DIR, f'dataset_{num}2020_zeroed_v2')
        
        if not os.path.exists(source_path):
            print(f"⚠️ 找不到來源資料夾：{source_path}，已跳過。")
            continue
            
        process_and_save_data(source_path, target_path, num)
        
    print("\n🏁 所有資料集皆已完成「相對零點校正 (First Row Subtraction)」！")