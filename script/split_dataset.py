"""
数据集切分脚本
将用户-物品交互序列按比例切分成训练集、验证集、测试集
并保存稀疏矩阵
"""
import numpy as np
import scipy.sparse as sp
import os
from collections import defaultdict


def load_interactions(interaction_file):
    """
    加载用户-物品交互文件
    
    Args:
        interaction_file: 交互文件路径 (格式: 用户ID 物品ID1 物品ID2 ...)
    
    Returns:
        dict: {用户ID: [物品ID列表]}
    """
    user_items = {}
    with open(interaction_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                user_id = int(parts[0])
                item_ids = [int(x) for x in parts[1:]]
                user_items[user_id] = item_ids
    return user_items


def split_user_interactions(user_items, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    对每个用户的交互序列进行切分
    
    Args:
        user_items: {用户ID: [物品ID列表]}
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    
    Returns:
        train_data, val_data, test_data: 三个数据集
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    train_data = {}
    val_data = {}
    test_data = {}
    
    for user_id, items in user_items.items():
        n_items = len(items)
        
        # 计算切分点
        train_end = int(n_items * train_ratio)
        val_end = train_end + int(n_items * val_ratio)
        
        # 确保每个集合至少有1个物品
        if train_end == 0:
            train_end = 1
        if val_end <= train_end:
            val_end = train_end + 1
        if val_end >= n_items:
            val_end = n_items - 1
        
        # 切分
        train_items = items[:train_end]
        val_items = items[train_end:val_end]
        test_items = items[val_end:]
        
        # 保存（至少有一个物品才保存）
        if len(train_items) > 0:
            train_data[user_id] = train_items
        if len(val_items) > 0:
            val_data[user_id] = val_items
        if len(test_items) > 0:
            test_data[user_id] = test_items
    
    return train_data, val_data, test_data


def save_interactions(data, output_file):
    """
    保存交互数据
    
    Args:
        data: {用户ID: [物品ID列表]}
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for user_id in sorted(data.keys()):
            items = data[user_id]
            f.write(f"{user_id} {' '.join(map(str, items))}\n")


def create_sparse_matrix(data, n_users, n_items):
    """
    创建稀疏矩阵 (用户-物品交互矩阵)
    
    Args:
        data: {用户ID: [物品ID列表]}
        n_users: 用户总数
        n_items: 物品总数
    
    Returns:
        scipy.sparse.csr_matrix: 稀疏矩阵 [n_users, n_items]
    """
    row = []
    col = []
    
    for user_id, items in data.items():
        for item_id in items:
            row.append(user_id)
            col.append(item_id)
    
    # 创建稀疏矩阵 (值全为1，表示有交互)
    data_values = np.ones(len(row), dtype=np.float32)
    sparse_matrix = sp.csr_matrix(
        (data_values, (row, col)), 
        shape=(n_users, n_items),
        dtype=np.float32
    )
    
    return sparse_matrix


def get_dataset_info(user_map_file, item_map_file):
    """
    获取用户和物品的总数
    
    Args:
        user_map_file: 用户ID映射文件
        item_map_file: 物品ID映射文件
    
    Returns:
        n_users, n_items
    """
    with open(user_map_file, 'r', encoding='utf-8') as f:
        n_users = sum(1 for _ in f)
    
    with open(item_map_file, 'r', encoding='utf-8') as f:
        n_items = sum(1 for _ in f)
    
    return n_users, n_items


def split_and_save(interaction_file, user_map_file, item_map_file, output_dir, 
                   train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    完整的切分和保存流程
    
    Args:
        interaction_file: 交互文件路径
        user_map_file: 用户ID映射文件
        item_map_file: 物品ID映射文件
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    print("=" * 60)
    print("数据集切分")
    print("=" * 60)
    
    # 1. 加载数据
    print(f"\n[1/5] 加载交互数据: {interaction_file}")
    user_items = load_interactions(interaction_file)
    print(f"   用户数: {len(user_items)}")
    total_interactions = sum(len(items) for items in user_items.values())
    print(f"   总交互数: {total_interactions}")
    
    # 2. 获取用户和物品总数
    print(f"\n[2/5] 获取数据集信息")
    n_users, n_items = get_dataset_info(user_map_file, item_map_file)
    print(f"   用户总数: {n_users}")
    print(f"   物品总数: {n_items}")
    
    # 3. 切分数据
    print(f"\n[3/5] 切分数据集 (训练:{train_ratio}, 验证:{val_ratio}, 测试:{test_ratio})")
    train_data, val_data, test_data = split_user_interactions(
        user_items, train_ratio, val_ratio, test_ratio
    )
    
    train_count = sum(len(items) for items in train_data.values())
    val_count = sum(len(items) for items in val_data.values())
    test_count = sum(len(items) for items in test_data.values())
    
    print(f"   训练集: {len(train_data)} 用户, {train_count} 交互 ({train_count/total_interactions*100:.2f}%)")
    print(f"   验证集: {len(val_data)} 用户, {val_count} 交互 ({val_count/total_interactions*100:.2f}%)")
    print(f"   测试集: {len(test_data)} 用户, {test_count} 交互 ({test_count/total_interactions*100:.2f}%)")
    
    # 4. 保存切分后的数据
    print(f"\n[4/5] 保存切分后的数据到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, 'train.txt')
    val_file = os.path.join(output_dir, 'val.txt')
    test_file = os.path.join(output_dir, 'test.txt')
    
    save_interactions(train_data, train_file)
    print(f"   ✓ 训练集: {train_file}")
    
    save_interactions(val_data, val_file)
    print(f"   ✓ 验证集: {val_file}")
    
    save_interactions(test_data, test_file)
    print(f"   ✓ 测试集: {test_file}")
    
    # 保存完整交互数据集
    full_file = os.path.join(output_dir, 'full.txt')
    save_interactions(user_items, full_file)
    print(f"   ✓ 完整数据集: {full_file}")
    
    # 5. 保存稀疏矩阵
    print(f"\n[5/5] 创建并保存稀疏矩阵")
    
    # 训练集稀疏矩阵
    train_matrix = create_sparse_matrix(train_data, n_users, n_items)
    train_matrix_file = os.path.join(output_dir, 'train_matrix.npz')
    sp.save_npz(train_matrix_file, train_matrix)
    print(f"   ✓ 训练集稀疏矩阵: {train_matrix_file}")
    print(f"     形状: {train_matrix.shape}, 非零元素: {train_matrix.nnz}, 稀疏度: {1-train_matrix.nnz/(n_users*n_items):.6f}")
    
    # 验证集稀疏矩阵
    val_matrix = create_sparse_matrix(val_data, n_users, n_items)
    val_matrix_file = os.path.join(output_dir, 'val_matrix.npz')
    sp.save_npz(val_matrix_file, val_matrix)
    print(f"   ✓ 验证集稀疏矩阵: {val_matrix_file}")
    print(f"     形状: {val_matrix.shape}, 非零元素: {val_matrix.nnz}, 稀疏度: {1-val_matrix.nnz/(n_users*n_items):.6f}")
    
    # 测试集稀疏矩阵
    test_matrix = create_sparse_matrix(test_data, n_users, n_items)
    test_matrix_file = os.path.join(output_dir, 'test_matrix.npz')
    sp.save_npz(test_matrix_file, test_matrix)
    print(f"   ✓ 测试集稀疏矩阵: {test_matrix_file}")
    print(f"     形状: {test_matrix.shape}, 非零元素: {test_matrix.nnz}, 稀疏度: {1-test_matrix.nnz/(n_users*n_items):.6f}")
    
    # 完整数据集稀疏矩阵
    full_matrix = create_sparse_matrix(user_items, n_users, n_items)
    full_matrix_file = os.path.join(output_dir, 'full_matrix.npz')
    sp.save_npz(full_matrix_file, full_matrix)
    print(f"   ✓ 完整数据集稀疏矩阵: {full_matrix_file}")
    print(f"     形状: {full_matrix.shape}, 非零元素: {full_matrix.nnz}, 稀疏度: {1-full_matrix.nnz/(n_users*n_items):.6f}")
    
    # 保存统计信息
    stats_file = os.path.join(output_dir, 'split_statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("数据集切分统计信息\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"用户总数: {n_users}\n")
        f.write(f"物品总数: {n_items}\n")
        f.write(f"总交互数: {total_interactions}\n\n")
        f.write(f"训练集:\n")
        f.write(f"  用户数: {len(train_data)}\n")
        f.write(f"  交互数: {train_count} ({train_count/total_interactions*100:.2f}%)\n")
        f.write(f"  平均每用户: {train_count/len(train_data):.2f}\n\n")
        f.write(f"验证集:\n")
        f.write(f"  用户数: {len(val_data)}\n")
        f.write(f"  交互数: {val_count} ({val_count/total_interactions*100:.2f}%)\n")
        f.write(f"  平均每用户: {val_count/len(val_data):.2f}\n\n")
        f.write(f"测试集:\n")
        f.write(f"  用户数: {len(test_data)}\n")
        f.write(f"  交互数: {test_count} ({test_count/total_interactions*100:.2f}%)\n")
        f.write(f"  平均每用户: {test_count/len(test_data):.2f}\n\n")
        f.write(f"稀疏矩阵:\n")
        f.write(f"  训练集矩阵形状: {train_matrix.shape}\n")
        f.write(f"  训练集非零元素: {train_matrix.nnz}\n")
        f.write(f"  训练集稀疏度: {1-train_matrix.nnz/(n_users*n_items):.6f}\n\n")
        f.write(f"  验证集矩阵形状: {val_matrix.shape}\n")
        f.write(f"  验证集非零元素: {val_matrix.nnz}\n")
        f.write(f"  验证集稀疏度: {1-val_matrix.nnz/(n_users*n_items):.6f}\n\n")
        f.write(f"  测试集矩阵形状: {test_matrix.shape}\n")
        f.write(f"  测试集非零元素: {test_matrix.nnz}\n")
        f.write(f"  测试集稀疏度: {1-test_matrix.nnz/(n_users*n_items):.6f}\n\n")
        f.write(f"完整数据集:\n")
        f.write(f"  用户数: {len(user_items)}\n")
        f.write(f"  交互数: {total_interactions}\n")
        f.write(f"  矩阵形状: {full_matrix.shape}\n")
        f.write(f"  非零元素: {full_matrix.nnz}\n")
        f.write(f"  稀疏度: {1-full_matrix.nnz/(n_users*n_items):.6f}\n")
    
    print(f"   ✓ 统计信息: {stats_file}")
    
    print("\n" + "=" * 60)
    print("数据集切分完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  1. train.txt - 训练集 (用户ID 物品ID1 物品ID2 ...)")
    print(f"  2. val.txt - 验证集 (用户ID 物品ID1 物品ID2 ...)")
    print(f"  3. test.txt - 测试集 (用户ID 物品ID1 物品ID2 ...)")
    print(f"  4. full.txt - 完整数据集 (用户ID 物品ID1 物品ID2 ...)")
    print(f"  5. train_matrix.npz - 训练集稀疏矩阵 (scipy sparse matrix)")
    print(f"  6. val_matrix.npz - 验证集稀疏矩阵 (scipy sparse matrix)")
    print(f"  7. test_matrix.npz - 测试集稀疏矩阵 (scipy sparse matrix)")
    print(f"  8. full_matrix.npz - 完整数据集稀疏矩阵 (scipy sparse matrix)")
    print(f"  9. split_statistics.txt - 切分统计信息")
    print()


if __name__ == '__main__':
    # 配置参数
    # DATA_DIR = 'dataset/toy/processed'
    # OUTPUT_DIR = 'dataset/toy/split'
    
    DATA_DIR = 'dataset/sports/processed'
    OUTPUT_DIR = 'dataset/sports/split'

    # 输入文件
    interaction_file = os.path.join(DATA_DIR, 'user_item_interactions.txt')
    user_map_file = os.path.join(DATA_DIR, 'user_id_map.txt')
    item_map_file = os.path.join(DATA_DIR, 'item_id_map.txt')
    
    # 切分比例
    TRAIN_RATIO = 0.7   # 训练集 70%
    VAL_RATIO = 0.15    # 验证集 15%
    TEST_RATIO = 0.15   # 测试集 15%
    
    # 执行切分
    split_and_save(
        interaction_file=interaction_file,
        user_map_file=user_map_file,
        item_map_file=item_map_file,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )
    
