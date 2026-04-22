"""配置加载器"""
import os
import yaml
import pickle
def load_config(config_path='config.yml'):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # 设置设备
    if config['device'] == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda']
        import torch
        config['device'] = 'cuda' if torch.cuda.is_available else 'cpu'
    else:
        config['device'] = 'cpu'
    # 加载语义嵌入
    dataset_name = config['data']['name']
    data_dir = config['data'].get('data_dir', './data')
    usrprf_embeds_path = f"{data_dir}/{dataset_name}/usr_emb_np.pkl"
    itmprf_embeds_path = f"{data_dir}/{dataset_name}/itm_emb_np.pkl"
    with open(usrprf_embeds_path, 'rb') as f:
        config['usrprf_embeds'] = pickle.load(f)
    with open(itmprf_embeds_path, 'rb') as f:
        config['itmprf_embeds'] = pickle.load(f)
    # 初始化data中的user_num和item_num（会在load_data中设置）
    if 'user_num' not in config['data']:
        config['data']['user_num'] = None
    if 'item_num' not in config['data']:
        config['data']['item_num'] = None
    return config
