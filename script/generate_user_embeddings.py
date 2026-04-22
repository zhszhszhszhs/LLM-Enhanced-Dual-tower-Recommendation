"""
使用千问文本嵌入模型生成用户画像的向量表示
"""
import json
import numpy as np
import dashscope
from dashscope import TextEmbedding
from http import HTTPStatus
import time
from tqdm import tqdm
import pickle
import os

# 配置千问API
dashscope.api_key = "sk-4cad0db934e146968786dad4983e52ef"


def get_qwen_embedding(text, model='text-embedding-v4', expected_dim=1024, max_retries=3):
    """
    调用千问文本嵌入API
    
    Args:
        text: 输入文本
        model: 嵌入模型
               'text-embedding-v1': 1536维
               'text-embedding-v2': 1536维
               'text-embedding-v3': 1024维
               'text-embedding-v4': 1024维（推荐）
        expected_dim: 期望的嵌入维度
        max_retries: 最大重试次数
    
    Returns:
        numpy.array: 嵌入向量（保证是expected_dim维）
    """
    for attempt in range(max_retries):
        try:
            response = TextEmbedding.call(
                model=model,
                input=text
            )
            
            if response.status_code == HTTPStatus.OK:
                # 提取embedding
                embedding = response.output['embeddings'][0]['embedding']
                embedding_array = np.array(embedding, dtype=np.float32)
                
                # 检查维度是否正确
                if embedding_array.shape[0] != expected_dim:
                    print(f"警告: 返回的维度{embedding_array.shape[0]}与期望{expected_dim}不符")
                    # 如果维度不对，返回None而不是错误的向量
                    return None
                
                return embedding_array
            else:
                print(f"API调用失败: {response.code}, {response.message}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        except Exception as e:
            print(f"API调用异常 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    return None


def load_profiles(profile_file):
    """
    加载用户画像
    
    Args:
        profile_file: 用户画像JSON文件路径
    
    Returns:
        list: 用户画像列表
    """
    profiles = []
    with open(profile_file, 'r', encoding='utf-8') as f:
        for line in f:
            profile = json.loads(line.strip())
            profiles.append(profile)
    return profiles


def generate_embeddings(profile_file, output_file, model='text-embedding-v2', 
                       start_index=0, batch_save_interval=100, embedding_dim=1024):
    """
    批量生成用户画像的嵌入向量
    
    Args:
        profile_file: 用户画像文件
        output_file: 输出文件路径（.npy或.pkl）
        model: 嵌入模型
        start_index: 起始索引（用于断点续传）
        batch_save_interval: 每处理多少个保存一次（防止中断丢失）
        embedding_dim: 嵌入维度
    
    Returns:
        numpy.array: 所有嵌入向量 [n_users, embedding_dim]
    """
    print("=" * 60)
    print("生成用户画像嵌入向量")
    print("=" * 60)
    
    # 1. 加载用户画像
    print(f"\n[1/3] 加载用户画像: {profile_file}")
    profiles = load_profiles(profile_file)
    print(f"   用户总数: {len(profiles)}")
    
    # 检查是否有已存在的嵌入文件（用于断点续传）
    embeddings_list = []
    npy_file = output_file.replace('.pkl', '.npy')
    pkl_file = output_file.replace('.npy', '.pkl')
    
    if start_index > 0:
        print(f"   检测到起始索引 {start_index}，尝试加载已有嵌入...")
        loaded = False
        
        # 先尝试加载npy文件
        if os.path.exists(npy_file):
            try:
                existing = np.load(npy_file)
                embeddings_list = existing.tolist()
                print(f"   ✓ 从npy文件加载 {len(embeddings_list)} 个已有嵌入")
                loaded = True
            except Exception as e:
                print(f"   npy文件加载失败: {e}")
        
        # 如果npy失败，尝试pkl
        if not loaded and os.path.exists(pkl_file):
            try:
                with open(pkl_file, 'rb') as f:
                    embeddings_list = pickle.load(f)
                print(f"   ✓ 从pkl文件加载 {len(embeddings_list)} 个已有嵌入")
                loaded = True
            except Exception as e:
                print(f"   pkl文件加载失败: {e}")
        
        if not loaded:
            print(f"   ⚠️  无法加载已有嵌入，将从头开始")
            embeddings_list = []
            start_index = 0
        else:
            # 检查是否需要截断或补齐
            if len(embeddings_list) > start_index:
                embeddings_list = embeddings_list[:start_index]
                print(f"   截断到 {start_index} 个嵌入")
            elif len(embeddings_list) < start_index:
                print(f"   ⚠️  已有嵌入数({len(embeddings_list)})小于起始索引({start_index})")
                print(f"   调整起始索引为 {len(embeddings_list)}")
                start_index = len(embeddings_list)
    
    # 2. 生成嵌入
    print(f"\n[2/3] 生成嵌入向量")
    print(f"   模型: {model}")
    print(f"   起始索引: {start_index}")
    
    success_count = 0
    failed_count = 0
    failed_indices = []
    
    for i in tqdm(range(start_index, len(profiles)), desc="生成进度"):
        profile = profiles[i]
        user_index = profile.get('user_index', i)
        profile_data = profile.get('profile', {})
        summarization = profile_data.get('summarization', '')
        
        # 如果summarization为空或为'None'，使用零向量
        if not summarization or summarization == 'None':
            print(f"\n   警告: 用户{user_index}的画像为空，使用零向量")
            embedding = np.zeros(embedding_dim, dtype=np.float32)
            failed_indices.append(user_index)
        else:
            # 调用API生成嵌入
            embedding = get_qwen_embedding(summarization, model, expected_dim=embedding_dim)
            
            if embedding is None:
                print(f"\n   错误: 用户{user_index}的嵌入生成失败，使用零向量")
                embedding = np.zeros(embedding_dim, dtype=np.float32)
                failed_count += 1
                failed_indices.append(user_index)
            else:
                success_count += 1
        
        # 确保形状正确
        if embedding.shape[0] != embedding_dim:
            print(f"\n   错误: 用户{user_index}的嵌入维度错误({embedding.shape[0]}!={embedding_dim})，使用零向量")
            embedding = np.zeros(embedding_dim, dtype=np.float32)
            failed_count += 1
            failed_indices.append(user_index)
        
        embeddings_list.append(embedding)
        
        # 定期保存（防止中断丢失）
        if (i + 1) % batch_save_interval == 0:
            try:
                embeddings_array = np.array(embeddings_list, dtype=np.float32)
                np.save(output_file.replace('.pkl', '.npy'), embeddings_array)
                with open(output_file.replace('.npy', '.pkl'), 'wb') as f:
                    pickle.dump(embeddings_list, f)
            except Exception as e:
                print(f"\n   ⚠️  保存进度失败: {e}")
        
        # 避免API限流
        time.sleep(0.1)
    
    # 3. 保存最终结果
    print(f"\n[3/3] 保存嵌入向量")
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    
    # 保存为numpy格式
    np.save(output_file.replace('.pkl', '.npy'), embeddings_array)
    print(f"   ✓ numpy格式: {output_file.replace('.pkl', '.npy')}")
    print(f"     形状: {embeddings_array.shape}")
    
    # 也保存为pickle格式（方便加载列表）
    with open(output_file.replace('.npy', '.pkl'), 'wb') as f:
        pickle.dump(embeddings_list, f)
    print(f"   ✓ pickle格式: {output_file.replace('.npy', '.pkl')}")
    
    # 保存统计信息
    stats_file = output_file.replace('.npy', '_stats.txt').replace('.pkl', '_stats.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("用户嵌入向量统计信息\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"总用户数: {len(profiles)}\n")
        f.write(f"成功生成: {success_count}\n")
        f.write(f"失败/空值: {failed_count}\n")
        f.write(f"嵌入维度: {embeddings_array.shape[1]}\n")
        f.write(f"数据类型: {embeddings_array.dtype}\n")
        f.write(f"文件大小: {embeddings_array.nbytes / 1024 / 1024:.2f} MB\n\n")
        
        if failed_indices:
            f.write(f"失败的用户索引（前20个）:\n")
            for idx in failed_indices[:20]:
                f.write(f"  {idx}\n")
    
    print(f"   ✓ 统计信息: {stats_file}")
    
    print(f"\n完成！")
    print(f"   成功: {success_count}")
    print(f"   失败: {failed_count}")
    print(f"   嵌入形状: {embeddings_array.shape}")
    
    return embeddings_array


def test_single_embedding(profile_file, user_index=None):
    """
    测试单个用户画像的嵌入生成
    
    Args:
        profile_file: 用户画像文件
        user_index: 用户索引（None表示随机选择）
    """
    class Colors:
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        END = '\033[0m'
    
    # 加载画像
    profiles = load_profiles(profile_file)
    
    # 随机或指定选择
    if user_index is None:
        user_index = np.random.choice(len(profiles))
    
    profile = profiles[user_index]
    profile_data = profile.get('profile', {})
    summarization = profile_data.get('summarization', '')
    
    print(Colors.GREEN + "=" * 60 + Colors.END)
    print(Colors.GREEN + f"测试用户嵌入生成 (用户索引: {user_index})" + Colors.END)
    print(Colors.GREEN + "=" * 60 + Colors.END)
    
    print(Colors.YELLOW + "\n用户画像:\n" + Colors.END)
    print(summarization)
    
    print(Colors.YELLOW + "\n调用千问嵌入API中...\n" + Colors.END)
    embedding = get_qwen_embedding(summarization)
    
    if embedding is not None:
        print(Colors.GREEN + "生成的嵌入向量:\n" + Colors.END)
        print(f"   形状: {embedding.shape}")
        print(f"   数据类型: {embedding.dtype}")
        print(f"   向量范围: [{embedding.min():.4f}, {embedding.max():.4f}]")
        print(f"   向量范数: {np.linalg.norm(embedding):.4f}")
        print(f"   前10维: {embedding[:10]}")
    else:
        print(Colors.GREEN + "❌ 嵌入生成失败" + Colors.END)
    
    print(Colors.GREEN + "\n" + "=" * 60 + Colors.END)


if __name__ == '__main__':
    # 配置文件路径
    # PROFILE_FILE = 'dataset/toy/processed/user_profiles.json'
    # OUTPUT_FILE = 'dataset/toy/processed/user_embeddings.npy'
    
    PROFILE_FILE = 'dataset/sports/processed/user_profiles.json'
    OUTPUT_FILE = 'dataset/sports/processed/user_embeddings.npy'

    # 配置参数
    MODEL = 'text-embedding-v4'  # 千问嵌入模型
    EMBEDDING_DIM = 1024         # v3和v4是1024维，v1和v2是1536维
    START_INDEX = 0              # 起始索引（用于断点续传，从错误位置继续）
    BATCH_SAVE_INTERVAL = 100    # 每处理100个保存一次
    
    # 模式选择
    MODE = 'batch'  # 'test' 或 'batch'
    
    if MODE == 'test':
        # 测试模式：生成单个用户的嵌入
        print("测试模式：生成单个用户的嵌入\n")
        test_single_embedding(
            profile_file=PROFILE_FILE,
            user_index=None  # None表示随机选择
        )
    else:
        # 批量模式：生成所有用户的嵌入
        print("批量模式：生成所有用户的嵌入\n")
        
        # 显示配置
        print(f"配置信息:")
        print(f"  模型: {MODEL}")
        print(f"  嵌入维度: {EMBEDDING_DIM}")
        print(f"  起始索引: {START_INDEX}")
        print(f"  批次保存间隔: {BATCH_SAVE_INTERVAL}")
        print()
        
        embeddings = generate_embeddings(
            profile_file=PROFILE_FILE,
            output_file=OUTPUT_FILE,
            model=MODEL,
            start_index=START_INDEX,
            batch_save_interval=BATCH_SAVE_INTERVAL,
            embedding_dim=EMBEDDING_DIM
        )

