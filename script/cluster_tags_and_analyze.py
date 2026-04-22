"""
使用DBSCAN对标签进行聚类
分析物品类别和用户交互物品的类分布
"""
import numpy as np
import json
from sklearn.cluster import DBSCAN
from collections import defaultdict, Counter
import os


def load_tag_embeddings(embedding_file):
    """加载标签嵌入向量"""
    print(f"加载标签嵌入向量: {embedding_file}")
    embeddings = np.load(embedding_file)
    print(f"  形状: {embeddings.shape}")
    return embeddings


def load_tag_map(tag_map_file):
    """加载标签ID映射"""
    print(f"加载标签ID映射: {tag_map_file}")
    tag_map = {}
    with open(tag_map_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                tag_data = json.loads(line.strip())
                tag_id = tag_data.get('tag_id')
                tag = tag_data.get('tag', '')
                count = tag_data.get('count', 0)
                if tag_id is not None:
                    tag_map[tag_id] = {
                        'tag': tag,
                        'count': count
                    }
            except json.JSONDecodeError:
                continue
    print(f"  标签总数: {len(tag_map)}")
    return tag_map


def load_item_tag_ids(item_tag_ids_file):
    """加载物品标签ID映射"""
    print(f"加载物品标签ID映射: {item_tag_ids_file}")
    item_tags = {}
    with open(item_tag_ids_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item_data = json.loads(line.strip())
                item_id = item_data.get('item_index')
                tag_ids = item_data.get('tag_ids', [])
                if item_id is not None:
                    item_tags[item_id] = tag_ids
            except json.JSONDecodeError:
                continue
    print(f"  物品总数: {len(item_tags)}")
    return item_tags


def load_user_interactions(interaction_file):
    """加载用户交互数据"""
    print(f"加载用户交互数据: {interaction_file}")
    user_items = {}
    with open(interaction_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                user_id = int(parts[0])
                items = [int(x) for x in parts[1:]]
                user_items[user_id] = items
    print(f"  用户总数: {len(user_items)}")
    return user_items


def filter_high_frequency_tags(tag_map, max_count=None, percentile=99):
    """
    过滤出现次数过多的标签
    
    Args:
        tag_map: 标签映射字典
        max_count: 最大出现次数阈值（如果为None，使用percentile）
        percentile: 百分位数阈值（默认95，即过滤掉出现次数最多的5%标签）
    
    Returns:
        set: 被过滤掉的标签ID集合
    """
    print(f"\n过滤高频标签...")
    
    if max_count is None:
        # 使用百分位数
        counts = [info['count'] for info in tag_map.values()]
        threshold = np.percentile(counts, percentile)
        print(f"  使用百分位数阈值: {percentile}% = {threshold:.0f}")
    else:
        threshold = max_count
        print(f"  使用固定阈值: {threshold}")
    
    filtered_tags = set()
    for tag_id, info in tag_map.items():
        if info['count'] >= threshold:
            filtered_tags.add(tag_id)
    
    print(f"  过滤掉的标签数: {len(filtered_tags)}")
    print(f"  保留的标签数: {len(tag_map) - len(filtered_tags)}")
    
    return filtered_tags


def cluster_tags(embeddings, tag_map, filtered_tags, eps=0.3, min_samples=5):
    """
    使用DBSCAN对标签进行聚类
    
    Args:
        embeddings: 标签嵌入矩阵
        tag_map: 标签映射字典
        filtered_tags: 被过滤的标签ID集合
        eps: DBSCAN的eps参数
        min_samples: DBSCAN的min_samples参数
    
    Returns:
        tuple: (标签到类别的映射, 类别信息)
    """
    print(f"\n使用DBSCAN进行标签聚类...")
    print(f"  参数: eps={eps}, min_samples={min_samples}")
    
    # 获取保留的标签ID列表
    valid_tag_ids = [tag_id for tag_id in tag_map.keys() if tag_id not in filtered_tags]
    valid_tag_ids.sort()
    
    # 提取保留标签的嵌入向量
    valid_embeddings = embeddings[valid_tag_ids]
    
    print(f"  参与聚类的标签数: {len(valid_tag_ids)}")
    
    # DBSCAN聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    cluster_labels = dbscan.fit_predict(valid_embeddings)
    
    # 创建标签ID到类别的映射
    tag_to_cluster = {}
    for tag_id, cluster_id in zip(valid_tag_ids, cluster_labels):
        tag_to_cluster[tag_id] = int(cluster_id)
    
    # 统计类别信息
    cluster_info = {}
    for tag_id, cluster_id in zip(valid_tag_ids, cluster_labels):
        if cluster_id not in cluster_info:
            cluster_info[cluster_id] = {
                'tag_ids': [],
                'tag_names': [],
                'size': 0
            }
        cluster_info[cluster_id]['tag_ids'].append(tag_id)
        cluster_info[cluster_id]['tag_names'].append(tag_map[tag_id]['tag'])
        cluster_info[cluster_id]['size'] += 1
    
    # 统计结果
    n_clusters = len([c for c in cluster_labels if c != -1])  # 排除噪声点
    n_noise = list(cluster_labels).count(-1)
    
    print(f"  聚类结果:")
    print(f"    类别数: {n_clusters}")
    print(f"    噪声点: {n_noise}")
    print(f"    噪声比例: {n_noise/len(valid_tag_ids)*100:.2f}%")
    
    # 显示前10个最大的类别
    sorted_clusters = sorted(cluster_info.items(), key=lambda x: x[1]['size'], reverse=True)
    print(f"\n  前10个最大的类别:")
    for i, (cluster_id, info) in enumerate(sorted_clusters[:10], 1):
        if cluster_id != -1:  # 跳过噪声点
            print(f"    类别{cluster_id}: {info['size']}个标签")
            # 显示前3个标签名称作为示例
            sample_tags = info['tag_names'][:3]
            print(f"      示例标签: {', '.join(sample_tags)}")
    
    return tag_to_cluster, cluster_info


def assign_items_to_clusters(item_tags, tag_to_cluster):
    """
    确定每个物品属于哪些类
    
    Args:
        item_tags: 物品标签映射 {item_id: [tag_id1, tag_id2, ...]}
        tag_to_cluster: 标签到类别的映射 {tag_id: cluster_id}
    
    Returns:
        dict: {item_id: [cluster_id1, cluster_id2, ...]}
    """
    print(f"\n确定物品所属类别...")
    
    item_clusters = {}
    for item_id, tag_ids in item_tags.items():
        clusters = set()
        for tag_id in tag_ids:
            if tag_id in tag_to_cluster:
                cluster_id = tag_to_cluster[tag_id]
                if cluster_id != -1:  # 排除噪声点
                    clusters.add(cluster_id)
        item_clusters[item_id] = sorted(list(clusters))
    
    # 统计
    items_with_clusters = sum(1 for clusters in item_clusters.values() if clusters)
    print(f"  有类别的物品数: {items_with_clusters}")
    print(f"  无类别的物品数: {len(item_clusters) - items_with_clusters}")
    
    return item_clusters


def analyze_user_cluster_distribution(user_items, item_clusters):
    """
    分析用户交互物品的类分布
    
    Args:
        user_items: 用户交互数据 {user_id: [item_id1, item_id2, ...]}
        item_clusters: 物品类别映射 {item_id: [cluster_id1, cluster_id2, ...]}
    
    Returns:
        dict: 用户类分布统计
    """
    print(f"\n分析用户交互物品的类分布...")
    
    user_cluster_stats = {}
    
    for user_id, items in user_items.items():
        # 统计该用户交互物品的类别分布
        cluster_counts = Counter()
        items_in_clusters = 0
        
        for item_id in items:
            if item_id in item_clusters:
                clusters = item_clusters[item_id]
                if clusters:  # 如果物品有类别
                    items_in_clusters += 1
                    for cluster_id in clusters:
                        cluster_counts[cluster_id] += 1
        
        # 计算统计信息
        total_items = len(items)
        if total_items > 0:
            user_cluster_stats[user_id] = {
                'total_items': total_items,
                'items_in_clusters': items_in_clusters,
                'items_in_clusters_ratio': items_in_clusters / total_items,
                'unique_clusters': len(cluster_counts),
                'cluster_counts': dict(cluster_counts),
                'most_common_cluster': cluster_counts.most_common(1)[0][0] if cluster_counts else None,
                'most_common_cluster_count': cluster_counts.most_common(1)[0][1] if cluster_counts else 0
            }
    
    # 统计汇总
    total_users = len(user_cluster_stats)
    users_with_clusters = sum(1 for stats in user_cluster_stats.values() if stats['items_in_clusters'] > 0)
    avg_items_in_clusters = np.mean([stats['items_in_clusters'] for stats in user_cluster_stats.values()])
    avg_cluster_ratio = np.mean([stats['items_in_clusters_ratio'] for stats in user_cluster_stats.values()])
    avg_unique_clusters = np.mean([stats['unique_clusters'] for stats in user_cluster_stats.values()])
    
    print(f"  总用户数: {total_users}")
    print(f"  有类别物品的用户数: {users_with_clusters}")
    print(f"  平均每个用户的类别物品数: {avg_items_in_clusters:.2f}")
    print(f"  平均类别物品比例: {avg_cluster_ratio:.2%}")
    print(f"  平均每个用户涉及的类别数: {avg_unique_clusters:.2f}")
    
    return user_cluster_stats


def save_results(tag_to_cluster, cluster_info, item_clusters, user_cluster_stats, output_dir):
    """保存所有结果"""
    print(f"\n保存结果到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存标签到类别的映射
    tag_cluster_file = os.path.join(output_dir, 'tag_cluster_mapping.json')
    with open(tag_cluster_file, 'w', encoding='utf-8') as f:
        for tag_id, cluster_id in sorted(tag_to_cluster.items()):
            f.write(json.dumps({
                'tag_id': tag_id,
                'cluster_id': cluster_id
            }, ensure_ascii=False) + '\n')
    print(f"  ✓ 标签类别映射: {tag_cluster_file}")
    
    # 2. 保存类别信息
    cluster_info_file = os.path.join(output_dir, 'cluster_info.json')
    with open(cluster_info_file, 'w', encoding='utf-8') as f:
        for cluster_id, info in sorted(cluster_info.items()):
            if cluster_id != -1:  # 排除噪声点
                f.write(json.dumps({
                    'cluster_id': int(cluster_id),
                    'size': int(info['size']),
                    'tag_ids': [int(tid) for tid in info['tag_ids']],
                    'tag_names': info['tag_names']
                }, ensure_ascii=False) + '\n')
    print(f"  ✓ 类别信息: {cluster_info_file}")
    
    # 3. 保存物品类别映射
    item_cluster_file = os.path.join(output_dir, 'item_cluster_mapping.json')
    with open(item_cluster_file, 'w', encoding='utf-8') as f:
        for item_id, clusters in sorted(item_clusters.items()):
            f.write(json.dumps({
                'item_index': item_id,
                'cluster_ids': clusters
            }, ensure_ascii=False) + '\n')
    print(f"  ✓ 物品类别映射: {item_cluster_file}")
    
    # 4. 保存用户类分布统计
    user_stats_file = os.path.join(output_dir, 'user_cluster_statistics.json')
    with open(user_stats_file, 'w', encoding='utf-8') as f:
        for user_id, stats in sorted(user_cluster_stats.items()):
            # 转换numpy类型为Python原生类型
            stats_serializable = {
                'user_index': int(user_id),
                'total_items': int(stats['total_items']),
                'items_in_clusters': int(stats['items_in_clusters']),
                'items_in_clusters_ratio': float(stats['items_in_clusters_ratio']),
                'unique_clusters': int(stats['unique_clusters']),
                'cluster_counts': {int(k): int(v) for k, v in stats['cluster_counts'].items()},
                'most_common_cluster': int(stats['most_common_cluster']) if stats['most_common_cluster'] is not None else None,
                'most_common_cluster_count': int(stats['most_common_cluster_count'])
            }
            f.write(json.dumps(stats_serializable, ensure_ascii=False) + '\n')
    print(f"  ✓ 用户类分布统计: {user_stats_file}")
    
    # 5. 保存统计摘要
    summary_file = os.path.join(output_dir, 'clustering_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("标签聚类分析摘要\n")
        f.write("=" * 50 + "\n\n")
        
        n_clusters = len([c for c in cluster_info.keys() if c != -1])
        n_noise_tags = cluster_info.get(-1, {}).get('size', 0)
        
        f.write(f"聚类参数:\n")
        f.write(f"  eps: 0.3\n")
        f.write(f"  min_samples: 5\n\n")
        
        f.write(f"标签统计:\n")
        f.write(f"  总标签数: {len(tag_to_cluster)}\n")
        f.write(f"  类别数: {n_clusters}\n")
        f.write(f"  噪声标签数: {n_noise_tags}\n\n")
        
        f.write(f"物品统计:\n")
        items_with_clusters = sum(1 for clusters in item_clusters.values() if clusters)
        f.write(f"  总物品数: {len(item_clusters)}\n")
        f.write(f"  有类别的物品数: {items_with_clusters}\n")
        f.write(f"  无类别的物品数: {len(item_clusters) - items_with_clusters}\n\n")
        
        f.write(f"用户统计:\n")
        total_users = len(user_cluster_stats)
        users_with_clusters = sum(1 for stats in user_cluster_stats.values() if stats['items_in_clusters'] > 0)
        f.write(f"  总用户数: {total_users}\n")
        f.write(f"  有类别物品的用户数: {users_with_clusters}\n")
        if total_users > 0:
            avg_ratio = np.mean([stats['items_in_clusters_ratio'] for stats in user_cluster_stats.values()])
            f.write(f"  平均类别物品比例: {avg_ratio:.2%}\n")
    
    print(f"  ✓ 统计摘要: {summary_file}")


def main():
    """主函数"""
    print("=" * 70)
    print("标签聚类和用户交互分析")
    print("=" * 70)
    
    # 配置文件路径
    # EMBEDDING_FILE = 'dataset/toy/processed/tag_embeddings.npy'
    # TAG_MAP_FILE = 'dataset/toy/processed/item_tags_statistics_tag_id_map.json'
    # ITEM_TAG_IDS_FILE = 'dataset/toy/processed/item_tag_ids.json'
    # INTERACTION_FILE = 'dataset/toy/processed/user_item_interactions.txt'
    # OUTPUT_DIR = 'dataset/toy/processed/tag_clusters'

    EMBEDDING_FILE = 'dataset/beauty/processed/tag_embeddings.npy'
    TAG_MAP_FILE = 'dataset/beauty/processed/item_tags_statistics_tag_id_map.json'
    ITEM_TAG_IDS_FILE = 'dataset/beauty/processed/item_tag_ids.json'
    INTERACTION_FILE = 'dataset/beauty/processed/user_item_interactions.txt'
    OUTPUT_DIR = 'dataset/beauty/processed/tag_clusters'
        
    # 配置参数
    MAX_TAG_COUNT = None  # None表示使用百分位数，或设置具体数值（如100）
    PERCENTILE = 99.5  # 过滤掉出现次数最多的5%标签
    EPS = 0.1  # DBSCAN的eps参数
    MIN_SAMPLES = 2  # DBSCAN的min_samples参数
    
    # 1. 加载数据
    print(f"\n[1/5] 加载数据")
    embeddings = load_tag_embeddings(EMBEDDING_FILE)
    tag_map = load_tag_map(TAG_MAP_FILE)
    item_tags = load_item_tag_ids(ITEM_TAG_IDS_FILE)
    user_items = load_user_interactions(INTERACTION_FILE)
    
    # 2. 过滤高频标签
    print(f"\n[2/5] 过滤高频标签")
    filtered_tags = filter_high_frequency_tags(tag_map, max_count=MAX_TAG_COUNT, percentile=PERCENTILE)
    
    # 3. DBSCAN聚类
    print(f"\n[3/5] DBSCAN聚类")
    tag_to_cluster, cluster_info = cluster_tags(embeddings, tag_map, filtered_tags, eps=EPS, min_samples=MIN_SAMPLES)
    
    # 4. 确定物品类别
    print(f"\n[4/5] 确定物品类别")
    item_clusters = assign_items_to_clusters(item_tags, tag_to_cluster)
    
    # 5. 分析用户类分布
    print(f"\n[5/5] 分析用户类分布")
    user_cluster_stats = analyze_user_cluster_distribution(user_items, item_clusters)
    
    # 6. 保存结果
    print(f"\n保存结果")
    save_results(tag_to_cluster, cluster_info, item_clusters, user_cluster_stats, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()

