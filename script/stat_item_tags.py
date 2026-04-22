"""
统计item_tags.json中所有标签的出现次数，并为每个标签分配ID
输出格式：每行一个物品的JSON，包含物品ID和标签ID列表
"""
import json
from collections import Counter
import os


def load_item_tags_with_items(tags_file):
    """
    加载物品标签文件，保留物品信息
    
    Args:
        tags_file: 标签JSON文件路径
    
    Returns:
        tuple: (items_data列表, 所有标签的列表)
    """
    items_data = []
    all_tags = []
    print(f"加载标签文件: {tags_file}")
    
    with open(tags_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item_data = json.loads(line.strip())
                item_id = item_data.get('item_index')
                tags = item_data.get('tags', [])
                
                # 过滤空标签
                valid_tags = [tag.strip() for tag in tags if tag and tag.strip()]
                
                if item_id is not None:
                    items_data.append({
                        'item_index': item_id,
                        'tags': valid_tags
                    })
                    all_tags.extend(valid_tags)
                
            except json.JSONDecodeError as e:
                print(f"  警告: 第{line_num}行JSON解析失败: {e}")
                continue
            except Exception as e:
                print(f"  警告: 第{line_num}行处理失败: {e}")
                continue
    
    print(f"  物品总数: {len(items_data)}")
    print(f"  总标签数: {len(all_tags)}")
    print(f"  唯一标签数: {len(set(all_tags))}")
    return items_data, all_tags


def count_tags(all_tags):
    """
    统计标签出现次数
    
    Args:
        all_tags: 所有标签的列表
    
    Returns:
        Counter: 标签计数结果
    """
    print("\n统计标签出现次数...")
    tag_counter = Counter(all_tags)
    return tag_counter


def create_tag_id_mapping(tag_counter):
    """
    为每个标签创建ID映射（从0开始）
    
    Args:
        tag_counter: 标签计数器
    
    Returns:
        tuple: (tag_to_id字典, id_to_tag字典)
    """
    # 按出现次数降序排序，然后按标签名称排序（保证稳定性）
    sorted_tags = sorted(tag_counter.items(), key=lambda x: (-x[1], x[0]))
    
    # 创建标签到ID的映射（从0开始）
    tag_to_id = {}
    id_to_tag = {}
    
    for tag_id, (tag, count) in enumerate(sorted_tags):
        tag_to_id[tag] = tag_id
        id_to_tag[tag_id] = tag
    
    print(f"\n创建标签ID映射:")
    print(f"  唯一标签数: {len(tag_to_id)}")
    print(f"  ID范围: 0 - {len(tag_to_id) - 1}")
    
    return tag_to_id, id_to_tag


def save_tag_statistics(tag_counter, tag_to_id, output_file):
    """
    保存标签统计结果（带ID）
    
    Args:
        tag_counter: 标签计数器
        tag_to_id: 标签到ID的映射
        output_file: 输出文件路径
    """
    print(f"\n保存标签统计结果: {output_file}")
    
    # 按出现次数降序排序
    sorted_tags = sorted(tag_counter.items(), key=lambda x: (-x[1], x[0]))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for tag, count in sorted_tags:
            tag_id = tag_to_id[tag]
            f.write(f"{tag}\t{count}\t{tag_id}\n")
    
    print(f"  已保存 {len(sorted_tags)} 个唯一标签的统计结果")
    
    # 显示前10个最常见的标签
    print(f"\n前10个最常见的标签:")
    for i, (tag, count) in enumerate(sorted_tags[:10], 1):
        tag_id = tag_to_id[tag]
        print(f"  {i}. [{tag_id}] {tag}: {count} 次")


def save_items_with_tag_ids(items_data, tag_to_id, output_file):
    """
    保存每个物品及其标签ID列表（JSON格式，每行一个物品）
    
    Args:
        items_data: 物品数据列表
        tag_to_id: 标签到ID的映射
        output_file: 输出文件路径
    """
    print(f"\n保存物品标签ID映射: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item_data in items_data:
            item_id = item_data['item_index']
            tags = item_data['tags']
            
            # 将标签转换为标签ID列表
            tag_ids = [tag_to_id[tag] for tag in tags if tag in tag_to_id]
            
            # 输出JSON格式，每行一个物品
            output_item = {
                'item_index': item_id,
                'tag_ids': tag_ids
            }
            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
    
    print(f"  已保存 {len(items_data)} 个物品的标签ID映射")


def generate_statistics(tags_file, output_file, items_output_file):
    """
    生成标签统计和物品标签ID映射
    
    Args:
        tags_file: 输入的标签文件路径
        output_file: 输出的统计文件路径
        items_output_file: 输出的物品标签ID文件路径
    """
    print("=" * 60)
    print("统计物品标签并分配ID")
    print("=" * 60)
    
    # 1. 加载标签和物品数据
    items_data, all_tags = load_item_tags_with_items(tags_file)
    
    if not all_tags:
        print("  错误: 没有找到任何标签！")
        return
    
    # 2. 统计标签
    tag_counter = count_tags(all_tags)
    
    # 3. 创建标签ID映射
    tag_to_id, id_to_tag = create_tag_id_mapping(tag_counter)
    
    # 4. 保存标签统计结果（带ID）
    save_tag_statistics(tag_counter, tag_to_id, output_file)
    
    # 5. 保存物品标签ID映射（JSON格式，每行一个物品）
    save_items_with_tag_ids(items_data, tag_to_id, items_output_file)
    
    # 6. 保存标签ID映射文件
    tag_id_map_file = output_file.replace('.txt', '_tag_id_map.json')
    with open(tag_id_map_file, 'w', encoding='utf-8') as f:
        for tag_id in sorted(id_to_tag.keys()):
            tag = id_to_tag[tag_id]
            count = tag_counter[tag]
            f.write(json.dumps({
                'tag_id': tag_id,
                'tag': tag,
                'count': count
            }, ensure_ascii=False) + '\n')
    print(f"\n标签ID映射文件: {tag_id_map_file}")
    
    # 7. 生成统计摘要
    stats_summary_file = output_file.replace('.txt', '_summary.txt')
    with open(stats_summary_file, 'w', encoding='utf-8') as f:
        f.write("物品标签统计摘要\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"总标签数: {len(all_tags)}\n")
        f.write(f"唯一标签数: {len(tag_counter)}\n")
        f.write(f"标签ID范围: 0 - {len(tag_to_id) - 1}\n")
        f.write(f"平均每个标签出现次数: {len(all_tags) / len(tag_counter):.2f}\n\n")
        
        f.write(f"出现次数分布:\n")
        count_distribution = Counter(tag_counter.values())
        sorted_counts = sorted(count_distribution.items(), key=lambda x: -x[0])
        for count, num_tags in sorted_counts[:20]:  # 显示前20个
            f.write(f"  出现{count}次: {num_tags}个标签\n")
    
    print(f"统计摘要: {stats_summary_file}")
    print("\n" + "=" * 60)
    print("统计完成！")
    print("=" * 60)


if __name__ == '__main__':
    # 配置文件路径
    # TAGS_FILE = 'dataset/toy/processed/item_tags.json'
    # OUTPUT_FILE = 'dataset/toy/processed/item_tags_statistics.txt'
    # ITEMS_OUTPUT_FILE = 'dataset/toy/processed/item_tag_ids.json'

    TAGS_FILE = 'dataset/beauty/processed/item_tags.json'
    OUTPUT_FILE = 'dataset/beauty/processed/item_tags_statistics.txt'
    ITEMS_OUTPUT_FILE = 'dataset/beauty/processed/item_tag_ids.json'    
    # 生成统计
    generate_statistics(TAGS_FILE, OUTPUT_FILE, ITEMS_OUTPUT_FILE)
    
    print("\n输出文件格式说明:")
    print(f"  1. {OUTPUT_FILE}: 标签统计（标签名称\\t出现次数\\t标签ID）")
    print(f"  2. {ITEMS_OUTPUT_FILE}: 物品标签ID映射（JSON格式，每行一个物品）")
    print(f"     格式: {{\"item_index\": X, \"tag_ids\": [id1, id2, ...]}}")
    print(f"  3. tag_id_map.json: 标签ID映射（JSON格式，每行一个标签）")
    print(f"     格式: {{\"tag_id\": X, \"tag\": \"标签名\", \"count\": 出现次数}}")

