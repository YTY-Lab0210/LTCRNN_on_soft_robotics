import os
import glob
import shutil
import random

# ================= 1. 路徑與實驗設定 =================
source_dir = r'C:\Users\HAO\Desktop\YTY_from_macbook\dataset\dataset_new_new_new' 
base_target_dir = r'C:\Users\HAO\Desktop\YTY_from_macbook\dataset_xx2020_new_new_new'      # 將產生 4 個資料夾在這個目錄下

# 定義四組實驗的訓練集比例 (相對於總資料量的比例)
experiments = {
    'dataset_602020': 0.6
    # 'dataset_302020': 0.3,  
    # 'dataset_102020': 0.1,      
    # 'dataset_52020': 0.05,
    # 'dataset_12020': 0.01,
}

# 驗證集與測試集比例「永遠鎖死」
val_ratio = 0.2
test_ratio = 0.2

# ================= 2. 建立所有目標資料夾 (🌟 修正：先徹底清空舊資料) =================
folders = ['training', 'validation', 'test']
for exp_name in experiments.keys():
    exp_dir = os.path.join(base_target_dir, exp_name)
    
    # 🌟 關鍵防呆：如果資料夾已存在，整個刪除重建，避免幽靈檔案殘留！
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
        
    for folder in folders:
        os.makedirs(os.path.join(exp_dir, folder), exist_ok=True)

# ================= 3. 掃描檔案並按類別分組 (🌟 修正：強制排序) =================
print("🔍 正在掃描原始資料夾...")
# 🌟 關鍵防呆：加上 sorted()，確保在 Windows/Mac 上每次讀取的順序都絕對一致
all_files = sorted(glob.glob(os.path.join(source_dir, "*.csv")))

class_files = {
    'ball': [],
    'small_ball': [],
    'cylinder': [],
    'cube': [],
    'rubik_cube': [],
    'bottle': [],
    'phone': [],
    'support': [],
    'mouse': [],
    'doll': []
}

for file in all_files:
    filename = os.path.basename(file)
    label = filename.rsplit('_', 1)[0]
    if label in class_files:
        class_files[label].append(file)

# ================= 4. 嚴謹的洗牌與分配機制 =================
random.seed(42) # 凍結宇宙時間
print("\n🔀 開始進行科學對照分配並複製檔案...")

for label, files in class_files.items():
    total_files = len(files)
    if total_files == 0:
        continue
        
    # 1. 徹底洗牌
    random.shuffle(files)
    
    # 2. 計算固定數量的 Val 和 Test (各 20%)
    val_count = int(total_files * val_ratio)
    test_count = int(total_files * test_ratio)
    
    # 3. 把 Val 和 Test 切在陣列的「最尾端」，確保它們永遠不動！
    val_files = files[-(val_count + test_count) : -test_count]
    test_files = files[-test_count : ]
    
    # 前面剩下的 60% 就是「最大訓練資源池」
    max_train_files = files[ : -(val_count + test_count)]
    
    # 4. 針對四個實驗分別處理訓練集
    for exp_name, train_ratio in experiments.items():
        # 計算這個實驗需要多少筆訓練資料
        train_count = int(total_files * train_ratio)
        
        # 從資源池裡，統一「從頭開始拿」
        # 這樣 30% 一定包含在 60% 裡面，6% 一定包含在 15% 裡面，完美控制變因！
        exp_train_files = max_train_files[:train_count]
        
        # 5. 複製到對應的實驗資料夾
        def copy_to_target(file_list, subfolder):
            for f in file_list:
                dest = os.path.join(base_target_dir, exp_name, subfolder, os.path.basename(f))
                shutil.copy2(f, dest)
                
        copy_to_target(exp_train_files, 'training')
        copy_to_target(val_files, 'validation')
        copy_to_target(test_files, 'test')

# ================= 5. 驗證結果 =================
print("\n🎉 四組實驗資料集分割完成！驗證各資料夾檔案數：")
for exp_name in experiments.keys():
    print(f"\n📊 實驗組：{exp_name}")
    for folder in folders:
        count = len(glob.glob(os.path.join(base_target_dir, exp_name, folder, "*.csv")))
        print(f"  - {folder}: {count} 個檔案")