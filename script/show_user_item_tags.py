"""
显示指定用户交互过的物品及其标签
"""
import json
import sys


def load_user_interactions(interaction_file):
    """
    加载用户-物品交互数据
    
    Args:
        interaction_file: 用户交互文件路径
    
    Returns:
        dict: {user_id: [item_id1, item_id2, ...]}
    """
    user_items = {}
    print(f"加载用户交互数据: {interaction_file}")
    
    with open(interaction_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                user_id = int(parts[0])
                items = [int(x) for x in parts[1:]]
                user_items[user_id] = items
    
    print(f"  用户总数: {len(user_items)}")
    return user_items


def load_item_tags(tags_file):
    """
    加载物品标签数据
    
    Args:
        tags_file: 物品标签文件路径
    
    Returns:
        dict: {item_id: [tag1, tag2, ...]}
    """
    item_tags_dict = {}
    print(f"加载物品标签数据: {tags_file}")
    
    with open(tags_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item_data = json.loads(line.strip())
                item_id = item_data.get('item_index')
                tags = item_data.get('tags', [])
                if item_id is not None:
                    item_tags_dict[item_id] = tags
            except json.JSONDecodeError:
                continue
    
    print(f"  物品标签总数: {len(item_tags_dict)}")
    return item_tags_dict


def show_user_item_tags(user_id, user_items, item_tags_dict, output_file=None):
    """
    显示用户交互过的物品及其标签
    
    Args:
        user_id: 用户ID
        user_items: 用户-物品交互字典
        item_tags_dict: 物品标签字典
        output_file: 输出文件路径（可选）
    """
    if user_id not in user_items:
        print(f"错误: 用户ID {user_id} 不存在！")
        return
    
    items = user_items[user_id]
    print(f"\n用户 {user_id} 交互过的物品数: {len(items)}")
    print("=" * 80)
    
    output_lines = []
    
    for item_id in items:
        tags = item_tags_dict.get(item_id, [])
        tags_str = ", ".join(tags) if tags else "无标签"
        
        # 显示格式：物品ID: 标签1, 标签2, ...
        line = f"物品{item_id}: {tags_str}"
        print(line)
        output_lines.append(line)
    
    # 如果指定了输出文件，保存结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"用户 {user_id} 交互过的物品标签\n")
            f.write("=" * 80 + "\n\n")
            for line in output_lines:
                f.write(line + "\n")
        print(f"\n结果已保存到: {output_file}")


def main():
    """
    主函数
    """
    # 配置文件路径
    INTERACTION_FILE = 'dataset/toy/processed/user_item_interactions.txt'
    TAGS_FILE = 'dataset/toy/processed/item_tags.json'
    
    # 获取用户ID（从命令行参数或交互输入）
    if len(sys.argv) > 1:
        try:
            user_id = int(sys.argv[1])
        except ValueError:
            print("错误: 用户ID必须是整数！")
            print(f"用法: python {sys.argv[0]} <用户ID> [输出文件]")
            sys.exit(1)
    else:
        user_id = input("请输入用户ID: ")
        try:
            user_id = int(user_id)
        except ValueError:
            print("错误: 用户ID必须是整数！")
            sys.exit(1)
    
    # 可选的输出文件
    output_file = None
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print("=" * 80)
    print("显示用户物品标签")
    print("=" * 80)
    
    # 1. 加载数据
    user_items = load_user_interactions(INTERACTION_FILE)
    item_tags_dict = load_item_tags(TAGS_FILE)
    
    # 2. 显示结果
    show_user_item_tags(user_id, user_items, item_tags_dict, output_file)
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

