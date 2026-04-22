"""
使用LLM API生成物品画像
基于物品信息和用户评论，生成物品的用户类型总结
支持千问（通义千问）API
"""
import json
import numpy as np
import os
import time
from tqdm import tqdm
import dashscope
from dashscope import Generation
from http import HTTPStatus

# 配置千问API
dashscope.api_key = "sk-4cad0db934e146968786dad4983e52ef"  # 请填入您的通义千问API密钥
# 获取密钥: https://dashscope.console.aliyun.com/apiKey


def get_qwen_response_w_system(prompt, system_prompt, model='qwen-turbo', max_retries=3):
    """
    调用千问API获取响应
    
    Args:
        prompt: 用户输入的prompt
        system_prompt: 系统指令
        model: 使用的模型 ('qwen-turbo', 'qwen-plus', 'qwen-max')
        max_retries: 最大重试次数
    
    Returns:
        str: 千问的响应
    """
    for attempt in range(max_retries):
        try:
            response = Generation.call(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                result_format='message',  # 返回message格式
                top_p=0.8,
                temperature=0.7,
                max_tokens=1500
            )
            
            if response.status_code == HTTPStatus.OK:
                # 提取响应内容
                result = response.output.choices[0]['message']['content']
                return result
            else:
                print(f"API调用失败: {response.code}, {response.message}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
        except Exception as e:
            print(f"API调用异常 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
    
    return None


def load_system_prompt(system_prompt_file):
    """
    加载系统提示词
    
    Args:
        system_prompt_file: 系统提示词文件路径
    
    Returns:
        str: 系统提示词内容
    """
    system_prompt = ""
    with open(system_prompt_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            system_prompt += line
    return system_prompt


def load_item_prompts(prompts_file):
    """
    加载物品prompts
    
    Args:
        prompts_file: prompts JSON文件路径
    
    Returns:
        list: prompts列表
    """
    prompts = []
    with open(prompts_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            item_data = json.loads(line.strip())
            prompts.append(item_data['prompt'])
    return prompts


def generate_item_profiles(prompts_file, system_prompt_file, output_file, 
                          sample_size=None, model='qwen-turbo', start_index=0, append_mode=False):
    """
    批量生成物品画像
    
    Args:
        prompts_file: 输入的prompts文件
        system_prompt_file: 系统提示词文件
        output_file: 输出文件路径
        sample_size: 采样数量（None表示处理全部）
        model: 使用的模型
        start_index: 起始索引（用于断点续传）
        append_mode: 是否追加模式（True=追加，False=覆盖）
    """
    print("=" * 60)
    print("生成物品画像")
    print("=" * 60)
    
    # 1. 加载系统提示词
    print(f"\n[1/4] 加载系统提示词: {system_prompt_file}")
    system_prompt = load_system_prompt(system_prompt_file)
    print(f"   系统提示词长度: {len(system_prompt)} 字符")
    
    # 2. 加载物品prompts
    print(f"\n[2/4] 加载物品prompts: {prompts_file}")
    all_prompts = load_item_prompts(prompts_file)
    print(f"   总物品数: {len(all_prompts)}")
    
    # 3. 选择处理范围
    print(f"\n[3/4] 选择处理范围")
    
    # 确定索引范围
    if sample_size and sample_size < len(all_prompts):
        end_index = min(start_index + sample_size, len(all_prompts))
        selected_indices = list(range(start_index, end_index))
        selected_prompts = [all_prompts[i] for i in selected_indices]
        print(f"   处理范围: {start_index} - {end_index-1} (共 {len(selected_indices)} 个物品)")
    else:
        selected_indices = list(range(start_index, len(all_prompts)))
        selected_prompts = [all_prompts[i] for i in selected_indices]
        print(f"   处理范围: {start_index} - {len(all_prompts)-1} (共 {len(selected_indices)} 个物品)")
    
    if len(selected_indices) == 0:
        print("   ⚠️  没有需要处理的物品！")
        return []
    
    # 4. 批量生成
    print(f"\n[4/4] 调用LLM API生成物品画像")
    print(f"   模型: {model}")
    print(f"   模式: {'追加模式' if append_mode else '覆盖模式'}")
    
    results = []
    success_count = 0
    failed_count = 0
    
    # 根据模式选择打开方式
    file_mode = 'a' if append_mode else 'w'
    
    with open(output_file, file_mode, encoding='utf-8') as f:
        for idx, prompt in tqdm(zip(selected_indices, selected_prompts), 
                                total=len(selected_prompts), 
                                desc="生成进度"):
            # 调用API
            response = get_qwen_response_w_system(prompt, system_prompt, model)
            
            if response:
                try:
                    # 尝试解析JSON响应
                    profile = json.loads(response)
                    result = {
                        'item_index': idx,
                        'profile': profile
                    }
                    success_count += 1
                except json.JSONDecodeError:
                    # 如果无法解析为JSON，保存原始响应
                    result = {
                        'item_index': idx,
                        'profile': {
                            'summarization': response,
                            'reasoning': 'Unable to parse as JSON'
                        }
                    }
                    success_count += 1
            else:
                # API调用失败
                result = {
                    'item_index': idx,
                    'profile': {
                        'summarization': 'None',
                        'reasoning': 'API call failed'
                    }
                }
                failed_count += 1
            
            # 保存结果
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            results.append(result)
            
            # 避免API限流
            time.sleep(0.5)
    
    print(f"\n完成！")
    print(f"   成功: {success_count}")
    print(f"   失败: {failed_count}")
    print(f"   输出文件: {output_file}")
    
    return results


def test_single_item(prompts_file, system_prompt_file, item_index=None, model='qwen-turbo'):
    """
    测试单个物品的生成效果
    
    Args:
        prompts_file: prompts文件路径
        system_prompt_file: 系统提示词文件路径
        item_index: 物品索引（None表示随机选择）
        model: 使用的模型
    """
    class Colors:
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        END = '\033[0m'
    
    # 加载数据
    system_prompt = load_system_prompt(system_prompt_file)
    prompts = load_item_prompts(prompts_file)
    
    # 随机或指定选择
    if item_index is None:
        item_index = np.random.choice(len(prompts))
    
    selected_prompt = prompts[item_index]
    
    # 显示信息
    print(Colors.GREEN + "=" * 60 + Colors.END)
    print(Colors.GREEN + f"测试物品画像生成 (物品索引: {item_index})" + Colors.END)
    print(Colors.GREEN + "=" * 60 + Colors.END)
    
    print(Colors.YELLOW + "\n系统提示词 (System Prompt):\n" + Colors.END)
    print(system_prompt)
    
    print(Colors.YELLOW + "\n输入Prompt:\n" + Colors.END)
    # 截断显示过长的prompt
    if len(selected_prompt) > 1000:
        print(selected_prompt[:1000] + "...\n(已截断)")
    else:
        print(selected_prompt)
    
    print(Colors.YELLOW + "\n调用千问API中...\n" + Colors.END)
    response = get_qwen_response_w_system(selected_prompt, system_prompt, model)
    
    print(Colors.GREEN + "生成的物品画像:\n" + Colors.END)
    if response:
        try:
            profile = json.loads(response)
            print(json.dumps(profile, ensure_ascii=False, indent=2))
        except:
            print(response)
    else:
        print("API调用失败")
    
    print(Colors.GREEN + "\n" + "=" * 60 + Colors.END)


if __name__ == '__main__':
    # 配置文件路径
    DATASET = 'sports'  # 可选: beauty / toys / sports
    SYSTEM_PROMPT_FILE = f'generation/item/item_system_prompt_{DATASET}.txt'
    PROMPTS_FILE = f'dataset/{DATASET}/processed/llm_input.json'
    OUTPUT_FILE = f'dataset/{DATASET}/processed/item_profiles.json'
    # 配置参数
    # 千问模型选项:
    # 'qwen-turbo': 快速、便宜（推荐）
    # 'qwen-plus': 更好的效果
    # 'qwen-max': 最强效果
    MODEL = 'qwen-turbo'
    SAMPLE_SIZE = None  # None表示处理全部，或设置为具体数字（如100）进行测试
    
    # 断点续传配置
    START_INDEX = 1891   # 起始索引，从0开始。如果中断了，设置为下一个要处理的索引
    APPEND_MODE = True  # True=追加到已有文件，False=覆盖文件
    
    # 模式选择
    MODE = 'batch'  # 'test' 或 'batch'
    
    if MODE == 'test':
        # 测试模式：生成单个物品画像
        print("测试模式：生成单个物品的画像\n")
        test_single_item(
            prompts_file=PROMPTS_FILE,
            system_prompt_file=SYSTEM_PROMPT_FILE,
            item_index=None,  # None表示随机选择
            model=MODEL
        )
    else:
        # 批量模式：生成所有物品画像
        print("批量模式：生成所有物品的画像\n")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        # 显示配置信息
        print(f"配置信息:")
        print(f"  起始索引: {START_INDEX}")
        print(f"  文件模式: {'追加' if APPEND_MODE else '覆盖'}")
        print(f"  模型: {MODEL}")
        
        # 如果是追加模式，检查已有文件
        if APPEND_MODE and os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                existing_count = sum(1 for _ in f)
            print(f"  已有记录: {existing_count} 条")
            print(f"  提示: 从索引 {START_INDEX} 开始追加\n")
        else:
            if not APPEND_MODE and os.path.exists(OUTPUT_FILE):
                print(f"  警告: 将覆盖已有文件 {OUTPUT_FILE}\n")
            else:
                print(f"  将创建新文件\n")
        
        # 生成物品画像
        results = generate_item_profiles(
            prompts_file=PROMPTS_FILE,
            system_prompt_file=SYSTEM_PROMPT_FILE,
            output_file=OUTPUT_FILE,
            sample_size=SAMPLE_SIZE,
            model=MODEL,
            start_index=START_INDEX,
            append_mode=APPEND_MODE
        )
        
        print("\n批量生成完成！")

