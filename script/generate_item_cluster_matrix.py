"""
生成物品-类别稀疏矩阵
形状: (item_num, cluster_num)
只保留物品数 <= 100 的类别
"""
import json
import numpy as np
import scipy.sparse as sp
import pickle
import os
from collections import defaultdict


def load_item_cluster_mapping(mapping_file):
    """加载物品-类别映射"""
    print(f"加载物品-类别映射: {mapping_file}")
    item_clusters = {}
    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                item_id = data.get('item_index')
                cluster_ids = data.get('cluster_ids', [])
                if item_id is not None:
                    item_clusters[item_id] = cluster_ids
            except json.JSONDecodeError:
                continue
    print(f"  物品总数: {len(item_clusters)}")
    return item_clusters


def load_cluster_info(cluster_info_file):
    """加载类别信息"""
    print(f"加载类别信息: {cluster_info_file}")
    cluster_info = {}
    with open(cluster_info_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                cluster_id = data.get('cluster_id')
                size = data.get('size', 0)
                if cluster_id is not None:
                    cluster_info[cluster_id] = size
            except json.JSONDecodeError:
                continue
    print(f"  类别总数: {len(cluster_info)}")
    return cluster_info


def filter_clusters_by_size(cluster_info, max_size=100):
    """过滤掉物品数 > max_size 的类别"""
    print(f"\n过滤类别（保留物品数 <= {max_size} 的类别）...")
    
    valid_clusters = {}
    filtered_out = []
    
    for cluster_id, size in cluster_info.items():
        if size <= max_size:
            valid_clusters[cluster_id] = size
        else:
            filtered_out.append((cluster_id, size))
    
    print(f"  保留的类别数: {len(valid_clusters)}")
    print(f"  过滤掉的类别数: {len(filtered_out)}")
    
    if filtered_out:
        print(f"\n  过滤掉的类别（物品数 > {max_size}）:")
        sorted_filtered = sorted(filtered_out, key=lambda x: x[1], reverse=True)
        for cluster_id, size in sorted_filtered[:10]:  # 只显示前10个
            print(f"    类别{cluster_id}: {size}个物品")
        if len(sorted_filtered) > 10:
            print(f"    ... 还有 {len(sorted_filtered) - 10} 个类别")
    
    return valid_clusters


def create_cluster_id_mapping(valid_clusters):
    """创建新的类别ID映射（从原始类别ID映射到新的连续ID，从0开始）"""
    print(f"\n创建类别ID映射...")
    
    # 按类别ID排序
    sorted_cluster_ids = sorted(valid_clusters.keys())
    
    # 创建映射：原始类别ID -> 新类别ID
    old_to_new = {}
    new_to_old = {}
    
    for new_id, old_id in enumerate(sorted_cluster_ids):
        old_to_new[old_id] = new_id
        new_to_old[new_id] = old_id
    
    print(f"  映射了 {len(old_to_new)} 个类别")
    print(f"  新类别ID范围: 0 - {len(old_to_new) - 1}")
    
    return old_to_new, new_to_old


def create_item_cluster_matrix(item_clusters, old_to_new_cluster, n_items):
    """
    创建物品-类别稀疏矩阵
    
    Args:
        item_clusters: {item_id: [cluster_id1, cluster_id2, ...]}
        old_to_new_cluster: {old_cluster_id: new_cluster_id}
        n_items: 物品总数
    
    Returns:
        scipy.sparse.csr_matrix: 稀疏矩阵 (n_items, n_clusters)
    """
    print(f"\n创建物品-类别稀疏矩阵...")
    
    n_clusters = len(old_to_new_cluster)
    row = []
    col = []
    
    # 统计
    items_with_clusters = 0
    total_assignments = 0
    
    for item_id, cluster_ids in item_clusters.items():
        # 只考虑有效的类别（在old_to_new_cluster中）
        valid_cluster_ids = [old_to_new_cluster[cid] for cid in cluster_ids if cid in old_to_new_cluster]
        
        if valid_cluster_ids:
            items_with_clusters += 1
            total_assignments += len(valid_cluster_ids)
            
            # 添加矩阵元素
            for cluster_id in valid_cluster_ids:
                row.append(item_id)
                col.append(cluster_id)
    
    print(f"  有类别的物品数: {items_with_clusters}")
    print(f"  总分配数: {total_assignments}")
    print(f"  平均每个物品的类别数: {total_assignments/items_with_clusters:.2f}")
    
    # 创建稀疏矩阵
    data_values = np.ones(len(row), dtype=np.float32)
    sparse_matrix = sp.csr_matrix(
        (data_values, (row, col)),
        shape=(n_items, n_clusters),
        dtype=np.float32
    )
    
    print(f"  矩阵形状: {sparse_matrix.shape}")
    print(f"  非零元素: {sparse_matrix.nnz}")
    print(f"  稀疏度: {1 - sparse_matrix.nnz / (n_items * n_clusters):.6f}")
    
    return sparse_matrix


def count_items_per_cluster(item_clusters, old_to_new_cluster):
    """统计每个新类别的物品数"""
    cluster_item_counts = defaultdict(int)
    
    for item_id, cluster_ids in item_clusters.items():
        for cluster_id in cluster_ids:
            if cluster_id in old_to_new_cluster:
                new_cluster_id = old_to_new_cluster[cluster_id]
                cluster_item_counts[new_cluster_id] += 1
    
    return dict(cluster_item_counts)


def save_matrix_and_mapping(sparse_matrix, old_to_new_cluster, new_to_old_cluster, 
                           cluster_item_counts, output_dir, max_size=100):
    """保存稀疏矩阵和映射信息"""
    print(f"\n保存结果到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存稀疏矩阵为pkl格式
    matrix_file = os.path.join(output_dir, f'item_cluster_matrix_max{max_size}.pkl')
    with open(matrix_file, 'wb') as f:
        pickle.dump(sparse_matrix, f)
    print(f"  ✓ 稀疏矩阵 (pkl): {matrix_file}")
    print(f"     形状: {sparse_matrix.shape}, 非零元素: {sparse_matrix.nnz}")
    
    # 2. 保存类别ID映射
    mapping_file = os.path.join(output_dir, f'cluster_id_mapping_max{max_size}.json')
    with open(mapping_file, 'w', encoding='utf-8') as f:
        # 保存为列表格式，方便查找
        mapping_data = {
            'old_to_new': old_to_new_cluster,
            'new_to_old': new_to_old_cluster
        }
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 类别ID映射: {mapping_file}")
    
    # 3. 保存类别统计信息
    stats_file = os.path.join(output_dir, f'cluster_statistics_max{max_size}.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        # 按物品数排序
        sorted_clusters = sorted(cluster_item_counts.items(), key=lambda x: x[1], reverse=True)
        stats_data = []
        for new_cluster_id, item_count in sorted_clusters:
            old_cluster_id = new_to_old_cluster[new_cluster_id]
            stats_data.append({
                'new_cluster_id': new_cluster_id,
                'old_cluster_id': old_cluster_id,
                'item_count': item_count
            })
        json.dump(stats_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 类别统计信息: {stats_file}")
    
    # 4. 保存摘要信息
    summary_file = os.path.join(output_dir, f'matrix_summary_max{max_size}.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("物品-类别稀疏矩阵统计摘要\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"过滤条件: 保留物品数 <= {max_size} 的类别\n\n")
        f.write(f"矩阵信息:\n")
        f.write(f"  形状: {sparse_matrix.shape}\n")
        f.write(f"  非零元素: {sparse_matrix.nnz}\n")
        f.write(f"  稀疏度: {1 - sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]):.6f}\n\n")
        f.write(f"类别信息:\n")
        f.write(f"  类别总数: {sparse_matrix.shape[1]}\n")
        f.write(f"  有类别的物品数: {len(set(sparse_matrix.nonzero()[0]))}\n\n")
        f.write(f"类别物品数分布:\n")
        sorted_counts = sorted(cluster_item_counts.values(), reverse=True)
        f.write(f"  最大: {max(sorted_counts)}\n")
        f.write(f"  最小: {min(sorted_counts)}\n")
        f.write(f"  平均: {np.mean(sorted_counts):.2f}\n")
        f.write(f"  中位数: {np.median(sorted_counts):.2f}\n")
    print(f"  ✓ 统计摘要: {summary_file}")


def main():
    """主函数"""
    print("=" * 70)
    print("生成物品-类别稀疏矩阵")
    print("=" * 70)
    
    # 配置文件路径
    # ITEM_CLUSTER_MAPPING_FILE = 'dataset/toy/processed/tag_clusters/item_cluster_mapping.json'
    # CLUSTER_INFO_FILE = 'dataset/toy/processed/tag_clusters/cluster_info.json'
    # ITEM_MAP_FILE = 'dataset/toy/processed/item_id_map.txt'
    # OUTPUT_DIR = 'dataset/toy/processed/tag_clusters'
    ITEM_CLUSTER_MAPPING_FILE = 'dataset/beauty/processed/tag_clusters/item_cluster_mapping.json'
    CLUSTER_INFO_FILE = 'dataset/beauty/processed/tag_clusters/cluster_info.json'
    ITEM_MAP_FILE = 'dataset/beauty/processed/item_id_map.txt'
    OUTPUT_DIR = 'dataset/beauty/processed/tag_clusters'

    # 配置参数
    MAX_CLUSTER_SIZE = 100  # 最大类别物品数
    
    # 1. 加载数据
    print(f"\n[1/5] 加载数据")
    item_clusters = load_item_cluster_mapping(ITEM_CLUSTER_MAPPING_FILE)
    cluster_info = load_cluster_info(CLUSTER_INFO_FILE)
    
    # 获取物品总数
    with open(ITEM_MAP_FILE, 'r', encoding='utf-8') as f:
        n_items = sum(1 for _ in f)
    print(f"  物品总数: {n_items}")
    
    # 2. 过滤类别
    print(f"\n[2/5] 过滤类别")
    valid_clusters = filter_clusters_by_size(cluster_info, max_size=MAX_CLUSTER_SIZE)
    
    # 3. 创建类别ID映射
    print(f"\n[3/5] 创建类别ID映射")
    old_to_new, new_to_old = create_cluster_id_mapping(valid_clusters)
    
    # 4. 创建稀疏矩阵
    print(f"\n[4/5] 创建稀疏矩阵")
    sparse_matrix = create_item_cluster_matrix(item_clusters, old_to_new, n_items)
    
    # 5. 统计类别物品数
    print(f"\n[5/5] 统计类别物品数")
    cluster_item_counts = count_items_per_cluster(item_clusters, old_to_new)
    print(f"  已统计 {len(cluster_item_counts)} 个类别的物品数")
    
    # 6. 保存结果
    print(f"\n保存结果")
    save_matrix_and_mapping(
        sparse_matrix, old_to_new, new_to_old, 
        cluster_item_counts, OUTPUT_DIR, MAX_CLUSTER_SIZE
    )
    
    print("\n" + "=" * 70)
    print("生成完成！")
    print("=" * 70)
    



if __name__ == '__main__':
    main()

