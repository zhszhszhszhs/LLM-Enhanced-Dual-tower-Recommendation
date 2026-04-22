"""
使用千问文本嵌入模型批量生成标签的向量表示
支持batch模式，一次编码多个标签
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


def get_qwen_embeddings_batch(texts, model='text-embedding-v4', expected_dim=1024, max_retries=3):
    """
    批量调用千问文本嵌入API
    
    Args:
        texts: 文本列表（可以是单个文本或文本列表）
        model: 嵌入模型
        expected_dim: 期望的嵌入维度
        max_retries: 最大重试次数
    
    Returns:
        list: 嵌入向量列表，每个元素是一个numpy.array
    """
    for attempt in range(max_retries):
        try:
            # 确保输入是列表格式
            if isinstance(texts, str):
                texts = [texts]
            
            response = TextEmbedding.call(
                model=model,
                input=texts  # batch模式：传入列表
            )
            
            if response.status_code == HTTPStatus.OK:
                embeddings = []
                output_embeddings = response.output.get('embeddings', [])
                
                for i, emb_data in enumerate(output_embeddings):
                    embedding = np.array(emb_data['embedding'], dtype=np.float32)
                    
                    # 检查维度是否正确
                    if embedding.shape[0] != expected_dim:
                        print(f"警告: 第{i}个文本返回的维度{embedding.shape[0]}与期望{expected_dim}不符，使用零向量")
                        embedding = np.zeros(expected_dim, dtype=np.float32)
                    
                    embeddings.append(embedding)
                
                return embeddings
            else:
                print(f"API调用失败: {response.code}, {response.message}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        except Exception as e:
            print(f"API调用异常 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    # 如果全部失败，返回零向量列表
    return [np.zeros(expected_dim, dtype=np.float32) for _ in texts]


def load_tag_map(tag_map_file):
    """
    加载标签ID映射文件
    
    Args:
        tag_map_file: 标签ID映射文件路径
    
    Returns:
        list: [(tag_id, tag), ...] 按tag_id排序
    """
    tags_data = []
    print(f"加载标签ID映射: {tag_map_file}")
    
    with open(tag_map_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                tag_data = json.loads(line.strip())
                tag_id = tag_data.get('tag_id')
                tag = tag_data.get('tag', '')
                if tag_id is not None:
                    tags_data.append((tag_id, tag))
            except json.JSONDecodeError:
                continue
    
    # 按tag_id排序
    tags_data.sort(key=lambda x: x[0])
    print(f"  标签总数: {len(tags_data)}")
    return tags_data


def generate_tag_embeddings(tag_map_file, output_file, model='text-embedding-v4', 
                            batch_size=25, embedding_dim=1024, start_index=0):
    """
    批量生成标签的嵌入向量
    
    Args:
        tag_map_file: 标签ID映射文件路径
        output_file: 输出文件路径（.npy）
        model: 嵌入模型
        batch_size: 每批处理的标签数量（建议10-50，根据API限制调整）
        embedding_dim: 嵌入维度
        start_index: 起始索引（用于断点续传）
    """
    print("=" * 60)
    print("批量生成标签嵌入向量")
    print("=" * 60)
    
    # 1. 加载标签数据
    print(f"\n[1/3] 加载标签数据: {tag_map_file}")
    tags_data = load_tag_map(tag_map_file)
    
    if not tags_data:
        print("  错误: 没有找到任何标签！")
        return None
    
    # 检查是否有已存在的嵌入文件（用于断点续传）
    embeddings_list = []
    npy_file = output_file.replace('.pkl', '.npy')
    
    if start_index > 0 and os.path.exists(npy_file):
        print(f"   检测到起始索引 {start_index}，尝试加载已有嵌入...")
        try:
            existing = np.load(npy_file)
            if existing.shape[1] == embedding_dim:
                embeddings_list = existing.tolist()
                print(f"   ✓ 从npy文件加载 {len(embeddings_list)} 个已有嵌入")
                if len(embeddings_list) > start_index:
                    embeddings_list = embeddings_list[:start_index]
                    print(f"   截断到 {start_index} 个嵌入")
                elif len(embeddings_list) < start_index:
                    print(f"   ⚠️  已有嵌入数({len(embeddings_list)})小于起始索引({start_index})")
                    print(f"   调整起始索引为 {len(embeddings_list)}")
                    start_index = len(embeddings_list)
            else:
                print(f"   ⚠️  已有文件维度 {existing.shape[1]} 与期望 {embedding_dim} 不符，将从头开始")
                embeddings_list = []
                start_index = 0
        except Exception as e:
            print(f"   加载失败: {e}，将从头开始")
            embeddings_list = []
            start_index = 0
    
    # 2. 批量生成嵌入
    print(f"\n[2/3] 批量生成嵌入向量")
    print(f"   模型: {model}")
    print(f"   批次大小: {batch_size}")
    print(f"   起始索引: {start_index}")
    print(f"   总标签数: {len(tags_data)}")
    
    success_count = 0
    failed_count = 0
    
    # 处理剩余标签
    remaining_tags = tags_data[start_index:]
    total_batches = (len(remaining_tags) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="生成进度"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(remaining_tags))
        batch_tags = remaining_tags[start:end]
        
        # 提取标签文本
        batch_texts = [tag for _, tag in batch_tags]
        batch_tag_ids = [tag_id for tag_id, _ in batch_tags]
        
        # 批量调用API
        batch_embeddings = get_qwen_embeddings_batch(
            batch_texts, 
            model=model, 
            expected_dim=embedding_dim
        )
        
        # 检查返回的嵌入数量
        if len(batch_embeddings) != len(batch_texts):
            print(f"\n   警告: 批次{batch_idx}返回的嵌入数({len(batch_embeddings)})与输入数({len(batch_texts)})不符")
            # 补齐缺失的嵌入
            while len(batch_embeddings) < len(batch_texts):
                batch_embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
            batch_embeddings = batch_embeddings[:len(batch_texts)]
        
        # 验证每个嵌入的维度
        for i, emb in enumerate(batch_embeddings):
            if emb.shape[0] != embedding_dim:
                print(f"\n   警告: 标签{batch_tag_ids[i]}的嵌入维度错误，使用零向量")
                batch_embeddings[i] = np.zeros(embedding_dim, dtype=np.float32)
                failed_count += 1
            else:
                success_count += 1
        
        embeddings_list.extend(batch_embeddings)
        
        # 定期保存（每10个批次保存一次）
        if (batch_idx + 1) % 10 == 0:
            try:
                embeddings_array = np.array(embeddings_list, dtype=np.float32)
                np.save(npy_file, embeddings_array)
            except Exception as e:
                print(f"\n   ⚠️  保存进度失败: {e}")
        
        # 避免API限流
        time.sleep(0.2)
    
    # 3. 保存最终结果
    print(f"\n[3/3] 保存嵌入向量")
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    
    # 保存为numpy格式
    np.save(npy_file, embeddings_array)
    print(f"   ✓ numpy格式: {npy_file}")
    print(f"     形状: {embeddings_array.shape}")
    print(f"     数据类型: {embeddings_array.dtype}")
    print(f"     文件大小: {embeddings_array.nbytes / 1024 / 1024:.2f} MB")
    
    # 保存统计信息
    stats_file = output_file.replace('.npy', '_stats.txt').replace('.pkl', '_stats.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("标签嵌入向量统计信息\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"总标签数: {len(tags_data)}\n")
        f.write(f"成功生成: {success_count}\n")
        f.write(f"失败/错误: {failed_count}\n")
        f.write(f"嵌入维度: {embeddings_array.shape[1]}\n")
        f.write(f"数据类型: {embeddings_array.dtype}\n")
        f.write(f"文件大小: {embeddings_array.nbytes / 1024 / 1024:.2f} MB\n")
    
    print(f"   ✓ 统计信息: {stats_file}")
    
    print(f"\n完成！")
    print(f"   成功: {success_count}")
    print(f"   失败: {failed_count}")
    print(f"   嵌入形状: {embeddings_array.shape}")
    
    return embeddings_array


def test_batch_embedding():
    """
    测试批量编码功能
    """
    class Colors:
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        END = '\033[0m'
    
    print(Colors.GREEN + "=" * 60 + Colors.END)
    print(Colors.GREEN + "测试批量标签编码" + Colors.END)
    print(Colors.GREEN + "=" * 60 + Colors.END)
    
    # 测试文本
    test_tags = [
        "Educational Play",
        "Durable Construction",
        "Child-Friendly Design",
        "Interactive Play",
        "STEM Learning"
    ]
    
    print(Colors.YELLOW + f"\n测试标签 ({len(test_tags)}个):" + Colors.END)
    for i, tag in enumerate(test_tags, 1):
        print(f"  {i}. {tag}")
    
    print(Colors.YELLOW + "\n调用千问批量嵌入API中...\n" + Colors.END)
    embeddings = get_qwen_embeddings_batch(test_tags, model='text-embedding-v4', expected_dim=1024)
    
    if embeddings:
        print(Colors.GREEN + "批量编码成功！\n" + Colors.END)
        for i, (tag, emb) in enumerate(zip(test_tags, embeddings), 1):
            print(f"标签 {i}: {tag}")
            print(f"  形状: {emb.shape}")
            print(f"  向量范数: {np.linalg.norm(emb):.4f}")
            print(f"  前5维: {emb[:5]}")
            print()
    else:
        print(Colors.GREEN + "❌ 批量编码失败" + Colors.END)
    
    print(Colors.GREEN + "=" * 60 + Colors.END)


if __name__ == '__main__':
    # 配置文件路径
    # TAG_MAP_FILE = 'dataset/toy/processed/item_tags_statistics_tag_id_map.json'
    # OUTPUT_FILE = 'dataset/toy/processed/tag_embeddings.npy'
    
    TAG_MAP_FILE = 'dataset/beauty/processed/item_tags_statistics_tag_id_map.json'
    OUTPUT_FILE = 'dataset/beauty/processed/tag_embeddings.npy'
    # 配置参数
    MODEL = 'text-embedding-v4'  # 千问嵌入模型
    EMBEDDING_DIM = 1024         # v4是1024维
    BATCH_SIZE = 10              # 每批处理的标签数量（建议10-50）
    START_INDEX = 0              # 起始索引（用于断点续传）
    
    # 模式选择
    MODE = 'batch'  # 'test' 或 'batch'
    
    if MODE == 'test':
        # 测试模式：测试批量编码功能
        print("测试模式：测试批量标签编码\n")
        test_batch_embedding()
    else:
        # 批量模式：生成所有标签的嵌入
        print("批量模式：生成所有标签的嵌入\n")
        
        # 显示配置
        print(f"配置信息:")
        print(f"  模型: {MODEL}")
        print(f"  嵌入维度: {EMBEDDING_DIM}")
        print(f"  批次大小: {BATCH_SIZE}")
        print(f"  起始索引: {START_INDEX}")
        print()
        
        embeddings = generate_tag_embeddings(
            tag_map_file=TAG_MAP_FILE,
            output_file=OUTPUT_FILE,
            model=MODEL,
            batch_size=BATCH_SIZE,
            embedding_dim=EMBEDDING_DIM,
            start_index=START_INDEX
        )

