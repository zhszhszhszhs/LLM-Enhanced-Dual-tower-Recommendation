"""
生成LLM输入格式的数据
将物品信息和用户评论整合成prompt格式
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


def load_reviews_by_item(reviews_file):
    """
    加载评论并按物品分组
    
    Args:
        reviews_file: 评论JSON文件路径
    
    Returns:
        dict: {item_id: [review1, review2, ...]}
    """
    item_reviews = defaultdict(list)
    with open(reviews_file, 'r', encoding='utf-8') as f:
        for line in f:
            review = json.loads(line.strip())
            item_id = review.get('item_id')
            if item_id is not None:
                item_reviews[item_id].append(review)
    return item_reviews


def format_categories(categories):
    """
    格式化类别信息
    
    Args:
        categories: 类别列表
    
    Returns:
        str: 格式化后的类别字符串
    """
    if not categories or len(categories) == 0:
        return ""
    
    # 将嵌套列表展平并连接
    category_strs = []
    for cat_list in categories:
        if isinstance(cat_list, list):
            category_strs.append(", ".join(cat_list))
        else:
            category_strs.append(str(cat_list))
    
    return " | ".join(category_strs)


def create_prompt(item_info, reviews):
    """
    创建LLM输入的prompt
    
    Args:
        item_info: 物品信息字典
        reviews: 评论列表
    
    Returns:
        str: 格式化的prompt
    """
    # 构建基本信息（不包含brand和categories）
    basic_info = {
        "item_id": item_info.get('item_id'),
        "title": item_info.get('title', ''),
        "description": item_info.get('description', '')
    }
    
    # 转换为JSON字符串
    basic_info_str = json.dumps(basic_info, ensure_ascii=False)
    
    # 构建用户反馈
    feedback_list = []
    for review in reviews:
        # 合并summary和reviewText
        summary = review.get('summary', '').strip()
        review_text = review.get('reviewText', '').strip()
        
        # 如果summary和reviewText都存在且不同，合并它们
        if summary and review_text:
            if summary.lower() not in review_text.lower():
                feedback = f"{summary}. {review_text}"
            else:
                feedback = review_text
        elif review_text:
            feedback = review_text
        elif summary:
            feedback = summary
        else:
            continue
        
        feedback_list.append(feedback)
    
    # 转换为JSON数组字符串
    feedback_str = json.dumps(feedback_list, ensure_ascii=False, indent=0)
    
    # 组合成完整的prompt
    prompt = f"BASIC INFORMATION: \n{basic_info_str}\nUSER FEEDBACK: \n{feedback_str}"
    
    return prompt


def generate_llm_input(items_file, reviews_file, output_file, min_reviews=1, max_reviews=10):
    """
    生成LLM输入格式的数据
    
    Args:
        items_file: 物品JSON文件路径
        reviews_file: 评论JSON文件路径
        output_file: 输出文件路径
        min_reviews: 最少评论数（过滤掉评论太少的物品）
        max_reviews: 最多评论数（限制每个物品的评论数量）
    """
    print("=" * 60)
    print("生成LLM输入格式数据")
    print("=" * 60)
    
    # 1. 加载物品信息
    print(f"\n[1/4] 加载物品信息: {items_file}")
    items = load_items(items_file)
    print(f"   物品总数: {len(items)}")
    
    # 2. 加载评论
    print(f"\n[2/4] 加载评论数据: {reviews_file}")
    item_reviews = load_reviews_by_item(reviews_file)
    print(f"   有评论的物品数: {len(item_reviews)}")
    total_reviews = sum(len(reviews) for reviews in item_reviews.values())
    print(f"   总评论数: {total_reviews}")
    
    # 3. 生成prompt格式数据
    print(f"\n[3/4] 生成prompt格式数据")
    print(f"   最少评论数: {min_reviews}")
    print(f"   最多评论数: {max_reviews}")
    
    saved_count = 0
    skipped_count = 0
    truncated_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 按item_id排序
        for item_id in sorted(items.keys()):
            item_info = items[item_id]
            reviews = item_reviews.get(item_id, [])
            original_review_count = len(reviews)
            
            # 过滤评论太少的物品
            if len(reviews) < min_reviews:
                skipped_count += 1
                continue
            
            # 限制评论数量
            if len(reviews) > max_reviews:
                reviews = reviews[:max_reviews]  # 只保留前max_reviews条
                truncated_count += 1
            
            # 创建prompt
            prompt = create_prompt(item_info, reviews)
            
            # 保存为JSON格式
            output_data = {"prompt": prompt}
            f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
            saved_count += 1
    
    print(f"   ✓ 已保存 {saved_count} 个物品")
    print(f"   ✓ 跳过 {skipped_count} 个物品（评论数 < {min_reviews}）")
    print(f"   ✓ 截断 {truncated_count} 个物品的评论（评论数 > {max_reviews}）")
    
    # 4. 生成统计信息
    print(f"\n[4/4] 保存统计信息")
    
    # 统计评论数分布
    review_counts = [len(reviews) for reviews in item_reviews.values()]
    review_counts.sort()
    
    stats_file = output_file.replace('.json', '_stats.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("LLM输入数据统计信息\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"物品总数: {len(items)}\n")
        f.write(f"有评论的物品数: {len(item_reviews)}\n")
        f.write(f"保存的物品数: {saved_count}\n")
        f.write(f"跳过的物品数: {skipped_count}\n")
        f.write(f"总评论数: {total_reviews}\n")
        f.write(f"平均每物品评论数: {total_reviews / len(item_reviews):.2f}\n\n")
        
        f.write(f"评论数分布:\n")
        f.write(f"  最少: {min(review_counts) if review_counts else 0}\n")
        f.write(f"  最多: {max(review_counts) if review_counts else 0}\n")
        f.write(f"  中位数: {review_counts[len(review_counts)//2] if review_counts else 0}\n")
        f.write(f"  25分位数: {review_counts[len(review_counts)//4] if review_counts else 0}\n")
        f.write(f"  75分位数: {review_counts[len(review_counts)*3//4] if review_counts else 0}\n")
    
    print(f"   ✓ 统计信息: {stats_file}")
    
    print("\n" + "=" * 60)
    print("生成完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  1. {os.path.basename(output_file)} - LLM输入格式数据")
    print(f"  2. {os.path.basename(stats_file)} - 统计信息")
    print()


def show_samples(output_file, num_samples=3):
    """
    显示样例数据
    
    Args:
        output_file: 输出文件路径
        num_samples: 显示的样例数
    """
    print(f"\n样例数据（前{num_samples}个物品）:")
    print("=" * 60)
    
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            
            data = json.loads(line.strip())
            prompt = data['prompt']
            
            # 截断显示（避免过长）
            if len(prompt) > 500:
                prompt = prompt[:500] + "..."
            
            print(f"\n物品 {i+1}:")
            print("-" * 60)
            print(prompt)
            print()


if __name__ == '__main__':
    # 配置参数
    # ITEMS_FILE = 'dataset/toy/processed/filtered_items.json'
    # REVIEWS_FILE = 'dataset/toy/processed/filtered_reviews.json'
    # OUTPUT_FILE = 'dataset/toy/processed/llm_input.json'

    ITEMS_FILE = 'dataset/sports/processed/filtered_items.json'
    REVIEWS_FILE = 'dataset/sports/processed/filtered_reviews.json'
    OUTPUT_FILE = 'dataset/sports/processed/llm_input.json'
    MIN_REVIEWS = 1   # 最少评论数
    MAX_REVIEWS = 10  # 最多评论数（控制输入长度）
    
    # 生成LLM输入数据
    generate_llm_input(
        items_file=ITEMS_FILE,
        reviews_file=REVIEWS_FILE,
        output_file=OUTPUT_FILE,
        min_reviews=MIN_REVIEWS,
        max_reviews=MAX_REVIEWS
    )
    
    # 显示样例
    show_samples(OUTPUT_FILE, num_samples=2)
    
