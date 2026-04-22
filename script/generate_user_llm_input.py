"""
生成用户画像的LLM输入格式数据
将用户购买过的物品信息和评论整合成prompt格式
"""
import json
from collections import defaultdict
import os


def load_items(items_file):
    """
    加载物品信息
    
    Args:
        items_file: 物品JSON文件路径
    
    Returns:
        dict: {item_id: item_info}
    """
    items = {}
    with open(items_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            item_id = item.get('item_id')
            if item_id is not None:
                items[item_id] = item
    return items


def load_reviews_by_user(reviews_file):
    """
    加载评论并按用户分组
    
    Args:
        reviews_file: 评论JSON文件路径
    
    Returns:
        dict: {user_id: [(item_id, review), ...]}
    """
    user_reviews = defaultdict(list)
    with open(reviews_file, 'r', encoding='utf-8') as f:
        for line in f:
            review = json.loads(line.strip())
            user_id = review.get('user_id')
            item_id = review.get('item_id')
            if user_id is not None and item_id is not None:
                user_reviews[user_id].append((item_id, review))
    return user_reviews


def format_review(review):
    """
    格式化用户评论，只使用reviewText
    
    Args:
        review: 评论字典
    
    Returns:
        str: 格式化后的评论文本
    """
    review_text = review.get('reviewText', '').strip()
    
    # 只返回reviewText，如果为空则返回"None"
    if review_text:
        return review_text
    else:
        return "None"


def create_prompt(user_items_data):
    """
    创建用户画像的LLM输入prompt
    
    Args:
        user_items_data: 用户购买物品列表，每个元素为 {"title": "...", "description": "...", "review": "..."}
    
    Returns:
        str: 格式化的prompt
    """
    # 转换为JSON数组字符串
    items_str = json.dumps(user_items_data, ensure_ascii=False, indent=0)
    
    # 组合成完整的prompt
    prompt = f"PURCHASED BUSINESSES: \n{items_str}"
    
    return prompt


def generate_user_llm_input(items_file, reviews_file, output_file, min_items=1, max_items=50):
    """
    生成用户画像的LLM输入格式数据
    
    Args:
        items_file: 物品JSON文件路径
        reviews_file: 评论JSON文件路径
        output_file: 输出文件路径
        min_items: 最少物品数（过滤掉购买太少的用户）
        max_items: 最多物品数（限制每个用户的物品数量）
    """
    print("=" * 60)
    print("生成用户画像LLM输入格式数据")
    print("=" * 60)
    
    # 1. 加载物品信息
    print(f"\n[1/4] 加载物品信息: {items_file}")
    items = load_items(items_file)
    print(f"   物品总数: {len(items)}")
    
    # 2. 加载评论并按用户分组
    print(f"\n[2/4] 加载评论数据: {reviews_file}")
    user_reviews = load_reviews_by_user(reviews_file)
    print(f"   有评论的用户数: {len(user_reviews)}")
    total_reviews = sum(len(reviews) for reviews in user_reviews.values())
    print(f"   总评论数: {total_reviews}")
    
    # 3. 生成prompt格式数据
    print(f"\n[3/4] 生成prompt格式数据")
    print(f"   最少物品数: {min_items}")
    print(f"   最多物品数: {max_items}")
    
    saved_count = 0
    skipped_count = 0
    truncated_count = 0
    missing_item_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 按user_id排序
        for user_id in sorted(user_reviews.keys()):
            reviews = user_reviews[user_id]
            original_item_count = len(reviews)
            
            # 过滤购买太少的用户
            if original_item_count < min_items:
                skipped_count += 1
                continue
            
            # 限制物品数量
            if original_item_count > max_items:
                reviews = reviews[:max_items]  # 只保留前max_items个
                truncated_count += 1
            
            # 构建用户购买物品列表
            user_items_data = []
            for item_id, review in reviews:
                # 获取物品信息
                item_info = items.get(item_id)
                if not item_info:
                    missing_item_count += 1
                    continue
                
                # 获取物品原始描述
                description = item_info.get('description', '').strip()
                if not description:
                    description = "None"
                
                # 格式化评论
                review_text = format_review(review)
                
                # 构建物品数据
                item_data = {
                    "title": item_info.get('title', 'None'),
                    "description": description,
                    "review": review_text
                }
                user_items_data.append(item_data)
            
            # 如果所有物品都缺失信息，跳过该用户
            if len(user_items_data) == 0:
                skipped_count += 1
                continue
            
            # 创建prompt
            prompt = create_prompt(user_items_data)
            
            # 保存为JSON格式
            output_data = {"prompt": prompt}
            f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
            saved_count += 1
    
    print(f"   ✓ 已保存 {saved_count} 个用户")
    print(f"   ✓ 跳过 {skipped_count} 个用户（物品数 < {min_items} 或缺少信息）")
    print(f"   ✓ 截断 {truncated_count} 个用户的物品（物品数 > {max_items}）")
    if missing_item_count > 0:
        print(f"   ⚠ 缺失物品信息: {missing_item_count} 次")
    
    # 4. 生成统计信息
    print(f"\n[4/4] 保存统计信息")
    
    # 统计物品数分布
    item_counts = [len(reviews) for reviews in user_reviews.values()]
    item_counts.sort()
    
    stats_file = output_file.replace('.json', '_stats.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("用户画像LLM输入数据统计信息\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"物品总数: {len(items)}\n")
        f.write(f"有评论的用户数: {len(user_reviews)}\n")
        f.write(f"保存的用户数: {saved_count}\n")
        f.write(f"跳过的用户数: {skipped_count}\n")
        f.write(f"总评论数: {total_reviews}\n")
        if len(user_reviews) > 0:
            f.write(f"平均每用户物品数: {total_reviews / len(user_reviews):.2f}\n\n")
        
        f.write(f"物品数分布:\n")
        if item_counts:
            f.write(f"  最少: {min(item_counts)}\n")
            f.write(f"  最多: {max(item_counts)}\n")
            f.write(f"  中位数: {item_counts[len(item_counts)//2]}\n")
            f.write(f"  25分位数: {item_counts[len(item_counts)//4]}\n")
            f.write(f"  75分位数: {item_counts[len(item_counts)*3//4]}\n")
        else:
            f.write(f"  最少: 0\n")
            f.write(f"  最多: 0\n")
            f.write(f"  中位数: 0\n")
            f.write(f"  25分位数: 0\n")
            f.write(f"  75分位数: 0\n")
    
    print(f"   ✓ 统计信息: {stats_file}")
    
    print("\n" + "=" * 60)
    print("生成完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  1. {os.path.basename(output_file)} - 用户画像LLM输入格式数据")
    print(f"  2. {os.path.basename(stats_file)} - 统计信息")
    print()


def show_samples(output_file, num_samples=3):
    """
    显示样例数据
    
    Args:
        output_file: 输出文件路径
        num_samples: 显示的样例数
    """
    print(f"\n样例数据（前{num_samples}个用户）:")
    print("=" * 60)
    
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            
            data = json.loads(line.strip())
            prompt = data['prompt']
            
            # 截断显示（避免过长）
            if len(prompt) > 800:
                prompt = prompt[:800] + "..."
            
            print(f"\n用户 {i}:")
            print("-" * 60)
            print(prompt)
            print()


if __name__ == '__main__':
    # 配置参数
    # ITEMS_FILE = 'dataset/toy/processed/filtered_items.json'
    # REVIEWS_FILE = 'dataset/toy/processed/filtered_reviews.json'
    # OUTPUT_FILE = 'dataset/toy/processed/user_llm_input.json'
    # MIN_ITEMS = 1    # 最少物品数
    # MAX_ITEMS = 50   # 最多物品数（控制输入长度）
    
    ITEMS_FILE = 'dataset/sports/processed/filtered_items.json'
    REVIEWS_FILE = 'dataset/sports/processed/filtered_reviews.json'
    OUTPUT_FILE = 'dataset/sports/processed/user_llm_input.json'
    MIN_ITEMS = 1    # 最少物品数
    MAX_ITEMS = 50   # 最多物品数（控制输入长度）

    # 生成用户画像LLM输入数据
    generate_user_llm_input(
        items_file=ITEMS_FILE,
        reviews_file=REVIEWS_FILE,
        output_file=OUTPUT_FILE,
        min_items=MIN_ITEMS,
        max_items=MAX_ITEMS
    )
    
    # 显示样例
    show_samples(OUTPUT_FILE, num_samples=2)
    
    print("\n使用示例:")
    print("```python")
    print("import json")
    print("")
    print("# 读取用户画像LLM输入数据")
    print("with open('dataset/toy/processed/user_llm_input.json', 'r', encoding='utf-8') as f:")
    print("    for line in f:")
    print("        data = json.loads(line.strip())")
    print("        prompt = data['prompt']")
    print("        ")
    print("        # 发送给LLM处理")
    print("        # response = llm.generate(prompt)")
    print("        # print(response)")
    print("```")

