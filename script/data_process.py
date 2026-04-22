"""
数据预处理脚本
处理Amazon评论数据，生成映射文件和交互文件
"""
import json
import pandas as pd
import os


def process_amazon_data(review_file, meta_file, output_dir, min_interactions=5):
    """
    处理Amazon评论数据
    
    Args:
        review_file: 评论数据文件路径
        meta_file: 商品元数据文件路径
        output_dir: 输出目录
        min_interactions: 最少交互次数（过滤用户和商品）
    
    生成的文件:
        - user_id_map.txt: 用户ID映射 (格式: 新ID 原始ID)
        - item_id_map.txt: 物品ID映射 (格式: 新ID 原始ID)
        - user_item_interactions.txt: 用户-物品交互 (格式: 用户ID 物品ID 评分 时间戳)
    """
    
    print("=" * 60)
    print("开始处理数据...")
    print("=" * 60)
    
    # 1. 加载商品元数据，筛选出有描述的商品
    print(f"\n[1/6] 加载商品元数据: {meta_file}")
    valid_items = set()
    item_info_dict = {}  # 保存商品完整信息
    total_items = 0
    with open(meta_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_items += 1
            try:
                item = eval(line.strip())  # meta文件使用Python dict格式
                asin = item.get('asin')
                description = item.get('description', '')
                
                # 只保留有描述且描述不为空的商品
                if asin and description and isinstance(description, str) and description.strip():
                    valid_items.add(asin)
                    item_info_dict[asin] = item  # 保存完整商品信息
            except:
                continue
    
    print(f"   总商品数: {total_items}")
    print(f"   有描述的商品数: {len(valid_items)}")
    print(f"   过滤比例: {len(valid_items)/total_items*100:.2f}%")
    
    # 2. 加载评论数据
    print(f"\n[2/6] 加载评论数据: {review_file}")
    data = []
    valid_reviews = []  # 保存满足条件的完整评论信息
    total_reviews = 0
    filtered_by_rating = 0
    filtered_by_text = 0
    filtered_by_item = 0
    
    with open(review_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_reviews += 1
            review = json.loads(line.strip())
            rating = review.get('overall', 0)
            review_text = review.get('reviewText', '')
            asin = review.get('asin', '')
            
            # 过滤条件1: 评分 >= 3
            if rating < 3.0:
                filtered_by_rating += 1
                continue
            
            # 过滤条件2: reviewText不为空
            if not review_text or not review_text.strip():
                filtered_by_text += 1
                continue
            
            # 过滤条件3: 商品必须有描述
            if asin not in valid_items:
                filtered_by_item += 1
                continue
            
            # 保存用于处理的数据
            data.append({
                'user_id': review['reviewerID'],
                'item_id': asin,
                'rating': rating,
                'timestamp': review['unixReviewTime']
            })
            
            # 保存完整的评论信息
            valid_reviews.append(review)
    
    df = pd.DataFrame(data)
    print(f"   总评论数: {total_reviews}")
    print(f"   过滤掉评分<3: {filtered_by_rating} ({filtered_by_rating/total_reviews*100:.2f}%)")
    print(f"   过滤掉空文本: {filtered_by_text} ({filtered_by_text/total_reviews*100:.2f}%)")
    print(f"   过滤掉无描述商品: {filtered_by_item} ({filtered_by_item/total_reviews*100:.2f}%)")
    print(f"   保留的评论数: {len(df)} ({len(df)/total_reviews*100:.2f}%)")
    print(f"   用户数: {df['user_id'].nunique()}")
    print(f"   物品数: {df['item_id'].nunique()}")
    
    # 3. 过滤低交互用户和商品
    print(f"\n[3/6] 过滤低交互用户和商品 (最少交互次数: {min_interactions})")
    while True:
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index
        
        df_filtered = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]
        
        if len(df_filtered) == len(df):
            break
        df = df_filtered
    
    print(f"   过滤后数据: {len(df)} 条交互")
    print(f"   过滤后用户数: {df['user_id'].nunique()}")
    print(f"   过滤后物品数: {df['item_id'].nunique()}")
    
    # 4. 创建ID映射
    print(f"\n[4/6] 创建ID映射")
    unique_users = sorted(df['user_id'].unique())
    unique_items = sorted(df['item_id'].unique())
    
    # 用户映射: 新ID -> 原始ID
    user_id_map = {idx: user for idx, user in enumerate(unique_users)}
    # 物品映射: 新ID -> 原始ID
    item_id_map = {idx: item for idx, item in enumerate(unique_items)}
    
    # 反向映射: 原始ID -> 新ID
    user_to_id = {user: idx for idx, user in user_id_map.items()}
    item_to_id = {item: idx for idx, item in item_id_map.items()}
    
    print(f"   用户映射: 0-{len(user_id_map)-1}")
    print(f"   物品映射: 0-{len(item_id_map)-1}")
    
    # 5. 生成交互数据
    print(f"\n[5/6] 生成交互数据")
    df['user_new_id'] = df['user_id'].map(user_to_id)
    df['item_new_id'] = df['item_id'].map(item_to_id)
    
    # 按时间排序
    df = df.sort_values('timestamp')
    
    # 6. 保存文件
    print(f"\n[6/6] 保存文件到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存用户ID映射
    user_map_file = os.path.join(output_dir, 'user_id_map.txt')
    with open(user_map_file, 'w', encoding='utf-8') as f:
        for new_id, orig_id in user_id_map.items():
            f.write(f"{new_id}\t{orig_id}\n")
    print(f"   ✓ 用户ID映射: {user_map_file}")
    
    # 保存物品ID映射
    item_map_file = os.path.join(output_dir, 'item_id_map.txt')
    with open(item_map_file, 'w', encoding='utf-8') as f:
        for new_id, orig_id in item_id_map.items():
            f.write(f"{new_id}\t{orig_id}\n")
    print(f"   ✓ 物品ID映射: {item_map_file}")
    
    # 保存用户-物品交互 (格式: 用户id 物品id1 物品id2 ...)
    interaction_file = os.path.join(output_dir, 'user_item_interactions.txt')
    
    # 按用户分组，每个用户一行
    user_items_dict = df.groupby('user_new_id')['item_new_id'].apply(list).to_dict()
    
    with open(interaction_file, 'w', encoding='utf-8') as f:
        for user_id in sorted(user_items_dict.keys()):
            items = user_items_dict[user_id]
            # 格式: 用户id 物品id1 物品id2 ...
            f.write(f"{user_id} {' '.join(map(str, items))}\n")
    
    print(f"   ✓ 用户-物品交互: {interaction_file}")
    
    # 保存满足条件的原始评论JSON（只保留文本相关信息，使用映射后的ID）
    reviews_json_file = os.path.join(output_dir, 'filtered_reviews.json')
    saved_reviews = 0
    skipped_reviews = 0
    
    with open(reviews_json_file, 'w', encoding='utf-8') as f:
        for review in valid_reviews:
            orig_user_id = review.get('reviewerID')
            orig_item_id = review.get('asin')
            
            # 检查用户和物品是否都在最终的映射中（低交互过滤后可能被移除）
            if orig_user_id not in user_to_id or orig_item_id not in item_to_id:
                skipped_reviews += 1
                continue
            
            # 只保留LLM生成用户画像所需的字段，使用映射后的ID
            cleaned_review = {
                'user_id': user_to_id[orig_user_id],  # 使用映射后的用户ID
                'item_id': item_to_id[orig_item_id],  # 使用映射后的物品ID
                'reviewText': review.get('reviewText', ''),
                'summary': review.get('summary', ''),
                'overall': review.get('overall'),
                'unixReviewTime': review.get('unixReviewTime')
            }
            f.write(json.dumps(cleaned_review, ensure_ascii=False) + '\n')
            saved_reviews += 1
    
    print(f"   ✓ 过滤后的评论JSON: {reviews_json_file}")
    print(f"     保留字段: user_id(映射后), item_id(映射后), reviewText, summary, overall, unixReviewTime")
    print(f"     保存评论数: {saved_reviews}, 跳过评论数: {skipped_reviews} (用户或物品被低交互过滤移除)")
    
    # 保存相关商品的原始信息JSON（只保留文本描述信息，使用映射后的ID）
    # 获取实际使用的商品集合（原始ID）
    used_items_orig = set(df['item_id'].unique())  # 这是原始asin
    items_json_file = os.path.join(output_dir, 'filtered_items.json')
    
    # 创建按映射后ID排序的商品列表
    item_list = []
    for orig_asin in used_items_orig:
        if orig_asin in item_info_dict:
            item = item_info_dict[orig_asin]
            new_item_id = item_to_id.get(orig_asin)
            item_list.append((new_item_id, item))
    
    # 按新ID排序
    item_list.sort(key=lambda x: x[0])
    
    with open(items_json_file, 'w', encoding='utf-8') as f:
        for new_item_id, item in item_list:
            # 只保留LLM生成用户画像所需的字段，使用映射后的ID
            cleaned_item = {
                'item_id': new_item_id,  # 使用映射后的物品ID
                'title': item.get('title', ''),
                'description': item.get('description', ''),
                'brand': item.get('brand', ''),
                'categories': item.get('categories', [])
            }
            f.write(json.dumps(cleaned_item, ensure_ascii=False) + '\n')
    print(f"   ✓ 过滤后的商品JSON: {items_json_file}")
    print(f"     保留字段: item_id(映射后), title, description, brand, categories")
    
    # 保存统计信息
    stats_file = os.path.join(output_dir, 'statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("数据集统计信息\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"用户数量: {len(user_id_map)}\n")
        f.write(f"物品数量: {len(item_id_map)}\n")
        f.write(f"交互数量: {len(df)}\n")
        f.write(f"稀疏度: {1 - len(df) / (len(user_id_map) * len(item_id_map)):.6f}\n")
        f.write(f"平均每用户交互数: {len(df) / len(user_id_map):.2f}\n")
        f.write(f"平均每物品交互数: {len(df) / len(item_id_map):.2f}\n")
        f.write(f"\n评分分布:\n")
        for rating, count in sorted(df['rating'].value_counts().items()):
            f.write(f"  评分 {rating}: {count} ({count/len(df)*100:.2f}%)\n")
    print(f"   ✓ 统计信息: {stats_file}")
    
    print("\n" + "=" * 60)
    print("数据处理完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  1. user_id_map.txt - 用户ID映射 (新ID -> 原始ID)")
    print(f"  2. item_id_map.txt - 物品ID映射 (新ID -> 原始ID)")
    print(f"  3. user_item_interactions.txt - 用户-物品交互 (用户ID 物品ID1 物品ID2 ...)")
    print(f"  4. filtered_reviews.json - 过滤后的评论原始数据 (JSON格式)")
    print(f"  5. filtered_items.json - 相关商品的原始数据 (JSON格式)")
    print(f"  6. statistics.txt - 数据集统计信息")
    print(f"\n数据集信息:")
    print(f"  用户数: {len(user_id_map)}")
    print(f"  物品数: {len(item_id_map)}")
    print(f"  交互数: {len(df)}")
    print(f"  稀疏度: {1 - len(df) / (len(user_id_map) * len(item_id_map)):.4%}")
    print()


if __name__ == '__main__':
    # 处理Toy数据集
    # process_amazon_data(
    #     review_file='dataset/toy/reviews_Toys_and_Games_5.json',
    #     meta_file='dataset/toy/meta_Toys_and_Games.json',
    #     output_dir='dataset/toy/processed',
    #     min_interactions=5
    # )
    
    # 如果需要处理Beauty数据集，取消下面的注释
    # process_amazon_data(
    #     review_file='dataset/beauty/reviews_Beauty_5.json',
    #     meta_file='dataset/beauty/meta_Beauty.json',
    #     output_dir='dataset/beauty/processed',
    #     min_interactions=5
    # )

    process_amazon_data(
        review_file='dataset/sports/reviews_Sports_and_Outdoors_5.json',
        meta_file='dataset/sports/meta_Sports_and_Outdoors.json',
        output_dir='dataset/sports/processed',
        min_interactions=6
    )    

