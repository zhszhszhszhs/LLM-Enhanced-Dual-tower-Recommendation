# 推荐系统数据处理工具

用于处理Amazon评论数据集的数据预处理和切分工具。

## 📁 项目结构

```
.
├── data_process.py          # 数据预处理脚本
├── split_dataset.py         # 数据集切分脚本
├── README.md               # 说明文档
└── dataset/
    └── toy/
        ├── reviews_Toys_and_Games_5.json  # 原始数据
        ├── processed/                      # 预处理后的数据
        │   ├── user_id_map.txt            # 用户ID映射
        │   ├── item_id_map.txt            # 物品ID映射
        │   ├── user_item_interactions.txt # 用户-物品交互
        │   └── statistics.txt             # 统计信息
        └── split/                          # 切分后的数据
            ├── train.txt                   # 训练集
            ├── val.txt                     # 验证集
            ├── test.txt                    # 测试集
            ├── train_matrix.npz            # 训练集稀疏矩阵
            ├── full_matrix.npz             # 完整数据稀疏矩阵
            └── split_statistics.txt        # 切分统计信息
```

## 🚀 使用流程

### 步骤1：数据预处理

```bash
python data_process.py
```

**功能：**
- 加载原始评论数据
- **只保留评分 >= 3 的交互**（隐式反馈）
- 过滤低交互用户和物品（默认最少5次交互）
- 将原始ID映射为连续的整数ID（从0开始）
- 生成用户ID映射、物品ID映射和交互文件

**生成文件：**

1. `user_id_map.txt` - 用户ID映射
   ```
   新ID    原始ID
   0       A1VXOAVRGKGEAK
   1       A8R62G708TSCM
   ...
   ```

2. `item_id_map.txt` - 物品ID映射
   ```
   新ID    原始ID
   0       0439893577
   1       B00000IZXW
   ...
   ```

3. `user_item_interactions.txt` - 用户-物品交互（每行一个用户）
   ```
   用户ID 物品ID1 物品ID2 物品ID3 ...
   0 125 340 567 892
   1 89 234 456
   2 45 123 678
   ...
   ```

4. `statistics.txt` - 数据集统计信息

### 步骤2：数据集切分

```bash
python split_dataset.py
```

**功能：**
- 加载预处理后的交互数据
- 对每个用户的物品序列按比例切分（默认70%训练，15%验证，15%测试）
- 生成训练集、验证集、测试集文件
- 创建稀疏矩阵（scipy格式）

**生成文件：**

1. `train.txt` - 训练集
   ```
   用户ID 物品ID1 物品ID2 ...
   0 125 340 567
   1 89 234
   ...
   ```

2. `val.txt` - 验证集
   ```
   用户ID 物品ID1 物品ID2 ...
   0 892
   1 456
   ...
   ```

3. `test.txt` - 测试集
   ```
   用户ID 物品ID1 物品ID2 ...
   0 1023 1145
   1 789 923
   ...
   ```

4. `train_matrix.npz` - 训练集稀疏矩阵（scipy sparse matrix）
5. `full_matrix.npz` - 完整数据稀疏矩阵（scipy sparse matrix）
6. `split_statistics.txt` - 切分统计信息

## 📊 数据格式说明

### 用户-物品交互文件格式
```
用户ID 物品ID1 物品ID2 物品ID3 ...
```
- 每行表示一个用户
- 第一个数字是用户ID
- 后续数字是该用户交互过的物品ID列表
- 物品按时间顺序排列（用于时序切分）

### 稀疏矩阵格式
- 使用 scipy 的 CSR 格式存储
- 形状: `[n_users, n_items]`
- 值为1表示有交互，0表示无交互

**加载稀疏矩阵：**
```python
import scipy.sparse as sp

# 加载训练集稀疏矩阵
train_matrix = sp.load_npz('dataset/toy/split/train_matrix.npz')
print(f"形状: {train_matrix.shape}")  # (n_users, n_items)
print(f"非零元素: {train_matrix.nnz}")

# 查看第0个用户的交互
user_0_interactions = train_matrix[0].toarray()
print(f"用户0交互的物品: {user_0_interactions}")

# 查看用户0交互过哪些物品
user_0_items = train_matrix[0].nonzero()[1]
print(f"用户0交互过的物品ID: {user_0_items}")
```

## ⚙️ 参数配置

### data_process.py 参数

在脚本中修改：
```python
process_amazon_data(
    review_file='dataset/toy/reviews_Toys_and_Games_5.json',  # 输入文件
    output_dir='dataset/toy/processed',                        # 输出目录
    min_interactions=5                                         # 最少交互次数
)
```

### split_dataset.py 参数

在脚本中修改：
```python
# 切分比例
TRAIN_RATIO = 0.7   # 训练集 70%
VAL_RATIO = 0.15    # 验证集 15%
TEST_RATIO = 0.15   # 测试集 15%
```

## 📦 依赖安装

```bash
pip install pandas numpy scipy
```

## 🎯 处理其他数据集

处理Beauty数据集，修改 `data_process.py`：

```python
process_amazon_data(
    review_file='dataset/beauty/reviews_Beauty_5.json',
    output_dir='dataset/beauty/processed',
    min_interactions=5
)
```

然后修改 `split_dataset.py`：

```python
DATA_DIR = 'dataset/beauty/processed'
OUTPUT_DIR = 'dataset/beauty/split'
```

## 💡 注意事项

1. **评分过滤**: 只保留评分 >= 3 的交互，转换为隐式反馈
2. **时序切分**: 按照交互的时间顺序进行切分，训练集在前，测试集在后
3. **最小交互数**: 过滤掉交互次数少于阈值的用户和物品，保证数据质量
4. **稀疏矩阵**: 使用scipy的CSR格式，节省内存空间

## 📈 示例输出

**数据预处理输出：**
```
==========================================================
开始处理数据...
==========================================================

[1/5] 加载数据: dataset/toy/reviews_Toys_and_Games_5.json
   原始数据: 145234 条交互 (评分 >= 3)
   原始用户数: 19412
   原始物品数: 11924

[2/5] 过滤数据 (最少交互次数: 5)
   过滤后数据: 126543 条交互
   过滤后用户数: 15529
   过滤后物品数: 9697

...
数据处理完成！
```

**数据集切分输出：**
```
==========================================================
数据集切分
==========================================================

[1/5] 加载交互数据: dataset/toy/processed/user_item_interactions.txt
   用户数: 15529
   总交互数: 126543

[3/5] 切分数据集 (训练:0.7, 验证:0.15, 测试:0.15)
   训练集: 15529 用户, 88580 交互 (70.00%)
   验证集: 15529 用户, 18982 交互 (15.00%)
   测试集: 15529 用户, 18981 交互 (15.00%)

...
数据集切分完成！
```


