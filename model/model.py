"""独立的 LightGCN Hypergraph 模型"""
import pickle
import torch as t
from torch import nn
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
from utils import cal_bpr_loss, reg_params, cal_infonce_loss, SpAdjEdgeDrop
init = nn.init.xavier_uniform_
class SimplifiedHypergraphConv(nn.Module):
    """简化超图卷积层
    实现公式: X^{(l+1)} = D^{-1} H B^{-1} H^{T} X^{(l)}
    其中:
    - H: 关联矩阵 (Item x Tag)，表示 Item 是否属于某个 Tag
    - B: 超边度矩阵 (Tag x Tag 对角矩阵)，表示每个 Tag 包含多少个 Item
    - D: 节点度矩阵 (Item x Item 对角矩阵)，表示每个 Item 属于多少个 Tag
    """
    def __init__(self, H, device):
        """
        Args:
            H: 关联矩阵，形状为 (item_num, tag_num)，可以是 numpy array 或 torch tensor
            device: 设备
        """
        super(SimplifiedHypergraphConv, self).__init__
        # 转换为 numpy array 以便计算
        if isinstance(H, t.Tensor):
            H_np = H.cpu.numpy
        else:
            H_np = np.array(H)
        # 确保 H 是二值矩阵（0 或 1）
        H_np = (H_np > 0).astype(np.float32)
        # 计算超边度矩阵 B (Tag 中包含多少个 Item)
        # B 是对角矩阵，B[i,i] = sum(H[:, i])
        B_diag = np.array(H_np.sum(axis=0)).flatten  # (tag_num,)
        # 避免除零，将 0 度设置为 1
        B_diag[B_diag == 0] = 1.0
        B_inv_diag = 1.0 / B_diag  # B^{-1} 的对角元素
        # 计算节点度矩阵 D (Item 属于多少个 Tag)
        # D 是对角矩阵，D[i,i] = sum(H[i, :])
        D_diag = np.array(H_np.sum(axis=1)).flatten  # (item_num,)
        # 避免除零，将 0 度设置为 1
        D_diag[D_diag == 0] = 1.0
        D_inv_diag = 1.0 / D_diag  # D^{-1} 的对角元素
        # 构建稀疏矩阵以便高效计算
        # 先计算 H B^{-1} H^{T}，然后左乘 D^{-1}
        H_sparse = sp.coo_matrix(H_np)
        # 构建 B^{-1} 对角矩阵
        tag_num = H_np.shape[1]
        B_inv_sparse = sp.diags(B_inv_diag, shape=(tag_num, tag_num))
        # 构建 D^{-1} 对角矩阵
        item_num = H_np.shape[0]
        D_inv_sparse = sp.diags(D_inv_diag, shape=(item_num, item_num))
        # 计算 D^{-1} H B^{-1} H^{T}
        # 步骤: H @ B^{-1} @ H^{T}，然后 D^{-1} @ (结果)
        temp = H_sparse @ B_inv_sparse @ H_sparse.T
        hypergraph_adj = D_inv_sparse @ temp
        # 转换为 torch sparse tensor
        hypergraph_adj = hypergraph_adj.tocoo
        idxs = t.from_numpy(np.vstack([hypergraph_adj.row, hypergraph_adj.col]).astype(np.int64))
        vals = t.from_numpy(hypergraph_adj.data.astype(np.float32))
        shape = t.Size(hypergraph_adj.shape)
        # 注册为 buffer（不参与梯度更新，但会随模型移动设备）
        self.register_buffer('hypergraph_adj', t.sparse.FloatTensor(idxs, vals, shape).to(device))
    def forward(self, item_embeds):
        """
        Args:
            item_embeds: Item 嵌入，形状为 (item_num, embedding_size)
        Returns:
            更新后的 Item 嵌入，形状为 (item_num, embedding_size)
        """
        # X^{(l+1)} = D^{-1} H B^{-1} H^{T} X^{(l)}
        return t.spmm(self.hypergraph_adj, item_embeds)
class LightGCN_hypergraph(nn.Module):
    """基于 LightGCN_plus 的模型，引入简化超图卷积层"""
    def __init__(self, config, data_handler):
        super(LightGCN_hypergraph, self).__init__
        # 基础参数（从BaseModel合并而来）
        self.config = config
        self.user_num = config['data']['user_num']
        self.item_num = config['data']['item_num']
        self.embedding_size = config['model']['embedding_size']
        # hyper-parameter
        if config['data']['name'] in config['model']:
            self.hyper_config = config['model'][config['data']['name']]
        else:
            self.hyper_config = config['model']
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']
        self.keep_rate = config['model']['keep_rate']
        # 模型组件
        self.adj = data_handler.torch_adj
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.edge_dropper = SpAdjEdgeDrop
        self.final_embeds = None
        self.is_training = False
        # semantic-embeddings
        # 确保是numpy array格式，避免警告
        if isinstance(config['usrprf_embeds'], np.ndarray):
            self.usrprf_embeds = t.tensor(config['usrprf_embeds']).float.cpu
        else:
            self.usrprf_embeds = t.tensor(np.array(config['usrprf_embeds'])).float.cpu
        if isinstance(config['itmprf_embeds'], np.ndarray):
            self.itmprf_embeds = t.tensor(config['itmprf_embeds']).float.cpu
        else:
            self.itmprf_embeds = t.tensor(np.array(config['itmprf_embeds'])).float.cpu
        self.mlp = nn.Sequential(
            nn.Linear(self.usrprf_embeds.shape[1], (self.usrprf_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU,
            nn.Linear((self.usrprf_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )
        # ========== 超图相关代码 ==========
        # 读取 hypergraph_weight（优先使用数据集特定配置，否则使用通用配置）
        dataset_name = config['data']['name']
        if dataset_name in config['model'] and 'hypergraph_weight' in config['model'][dataset_name]:
            # 使用数据集特定配置
            self.hypergraph_weight = float(config['model'][dataset_name]['hypergraph_weight'])
        elif 'hypergraph_weight' in config['model']:
            # 使用通用配置
            self.hypergraph_weight = float(config['model']['hypergraph_weight'])
        else:
            # 使用默认值
            self.hypergraph_weight = 0.0
        # 超图的嵌入（物品-意图超图）
        self.hypergraph_user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.hypergraph_item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.final_hypergraph_embeds = None
        # 加载 Item-Tag 关联矩阵 H 和用户-物品交互矩阵
        self._load_hypergraph_matrix(config, data_handler)
        self._load_user_item_matrix(data_handler, config)
        # 初始化超图卷积层
        self.hypergraph_conv = SimplifiedHypergraphConv(
            self.item_tag_matrix,
            config['device']
        )
        # ========== 超图相关代码结束 ==========
        self._init_weight
    def _init_weight(self):
        """初始化MLP权重"""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
    # ========== 超图相关方法 ==========
    def _load_hypergraph_matrix(self, config, data_handler):
        """加载预处理好的 Item-Tag 关联矩阵 H（稀疏矩阵格式）"""
        dataset_name = config['data']['name']
        data_dir = config['data']['data_dir']
        hypergraph_file = f'{data_dir}/{dataset_name}/hyper_matrices1.pkl'
        # 加载预处理好的超图矩阵（直接存储为稀疏矩阵）
        with open(hypergraph_file, 'rb') as f:
            H_sparse = pickle.load(f)
            # 检查是否为稀疏矩阵
            if isinstance(H_sparse, sp.spmatrix):
                # scipy 稀疏矩阵，转换为 numpy array（如果需要密集形式）
                self.item_tag_matrix = H_sparse.toarray.astype(np.float32)
            elif isinstance(H_sparse, t.Tensor):
                # torch tensor，转换为 numpy
                self.item_tag_matrix = H_sparse.cpu.numpy.astype(np.float32)
            else:
                # numpy array 或其他格式
                self.item_tag_matrix = np.array(H_sparse).astype(np.float32)
            # 验证维度：应该是 (item_num, tag_num)
            loaded_item_num, tag_num = self.item_tag_matrix.shape
            expected_item_num = self.item_num
            if loaded_item_num != expected_item_num:
                raise ValueError(
                    f"Hypergraph matrix dimension mismatch!\n"
                    f"  Expected shape: ({expected_item_num}, tag_num)\n"
                    f"  Loaded shape: ({loaded_item_num}, {tag_num})\n"
                    f"  This suggests the matrix might be (user_num, item_num) instead of (item_num, tag_num).\n"
                    f"  Please regenerate the hypergraph matrix with correct dimensions."
                )
    def _load_user_item_matrix(self, data_handler, config):
        """加载用户-物品交互矩阵，用于计算超图用户向量"""
        # 从 data_handler 获取训练矩阵
        trn_mat = data_handler.trn_mat  # scipy sparse matrix, shape: (user_num, item_num)
        # 转换为归一化的稀疏矩阵（按行归一化，用于加权平均）
        # 计算每个用户交互的物品数量
        user_degrees = np.array(trn_mat.sum(axis=1)).flatten  # (user_num,)
        user_degrees[user_degrees == 0] = 1.0  # 避免除零
        # 使用 scipy 的 diags 来高效构建归一化矩阵
        # 构建对角矩阵，每个元素是用户度的倒数
        user_degree_inv = sp.diags(1.0 / user_degrees, shape=(self.user_num, self.user_num))
        # 左乘对角矩阵实现按行归一化
        trn_mat_normalized = user_degree_inv @ trn_mat
        # 转换为 torch sparse tensor
        trn_mat_normalized = trn_mat_normalized.tocoo
        idxs = t.from_numpy(np.vstack([trn_mat_normalized.row, trn_mat_normalized.col]).astype(np.int64))
        vals = t.from_numpy(trn_mat_normalized.data.astype(np.float32))
        shape = t.Size(trn_mat_normalized.shape)
        # 注册为 buffer
        self.register_buffer('user_item_weight',
                           t.sparse.FloatTensor(idxs, vals, shape).to(config['device']))
    def _hypergraph_propagate(self, item_embeds):
        """超图卷积传播（只作用于物品）"""
        return self.hypergraph_conv(item_embeds)
    def _compute_hypergraph_user_embeds(self, hypergraph_item_embeds):
        """计算超图用户向量：原始用户向量 + 用户交互过的物品的超图向量加权和"""
        # user_item_weight 形状: (user_num, item_num)
        # hypergraph_item_embeds 形状: (item_num, embedding_size)
        # 交互物品向量加权和: (user_num, embedding_size)
        weighted_item_embeds = t.spmm(self.user_item_weight, hypergraph_item_embeds)
        # 原始用户向量 + 加权和
        return self.hypergraph_user_embeds + weighted_item_embeds
    # ========== 超图相关方法结束 ==========
    def _propagate(self, adj, embeds):
        """LightGCN传播"""
        return t.spmm(adj, embeds)
    def forward(self, adj=None, keep_rate=1.0):
        """前向传播"""
        if adj is None:
            adj = self.adj
        # 缓存检查
        if not self.is_training and self.final_embeds is not None and hasattr(self, 'final_hypergraph_embeds') and self.final_hypergraph_embeds is not None:
            return (self.final_embeds[:self.user_num], self.final_embeds[self.user_num:],
                    self.final_hypergraph_embeds[:self.user_num], self.final_hypergraph_embeds[self.user_num:])
        # LightGCN 传播
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]
        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
        self.final_embeds = embeds
        # ========== 超图相关代码 ==========
        # 超图物品嵌入：通过超图卷积传播
        hypergraph_item_embeds_list = [self.hypergraph_item_embeds]
        for i in range(self.layer_num):
            hypergraph_item_embeds = self._hypergraph_propagate(hypergraph_item_embeds_list[-1])
            hypergraph_item_embeds_list.append(hypergraph_item_embeds)
        hypergraph_item_embeds = sum(hypergraph_item_embeds_list)
        # 超图用户嵌入：通过交互物品的超图嵌入加权得到
        hypergraph_user_embeds = self._compute_hypergraph_user_embeds(hypergraph_item_embeds)
        # 缓存结果（仅在非训练模式下）
        if not self.is_training:
            self.final_hypergraph_embeds = t.concat([hypergraph_user_embeds, hypergraph_item_embeds], axis=0)
        return embeds[:self.user_num], embeds[self.user_num:], hypergraph_user_embeds, hypergraph_item_embeds
        # ========== 超图相关代码结束 ==========
    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        """从批次数据中提取嵌入"""
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds
    def _mask_predict(self, full_preds, train_mask):
        """屏蔽训练集中的正样本（从BaseModel合并而来）"""
        return full_preds * (1 - train_mask) - 1e8 * train_mask
    def cal_loss(self, batch_data):
        """计算损失"""
        self.is_training = True
        # ========== 超图相关代码 ==========
        # 计算超图嵌入
        user_embeds, item_embeds, hypergraph_user_embeds, hypergraph_item_embeds = self.forward(self.adj, self.keep_rate)
        # 使用 LightGCN 的嵌入计算 BPR 损失
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_embeds, item_embeds, batch_data)
        bpr_loss_lightgcn = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        # 使用超图的嵌入计算 BPR 损失
        anc_hypergraph_embeds, pos_hypergraph_embeds, neg_hypergraph_embeds = self._pick_embeds(
            hypergraph_user_embeds, hypergraph_item_embeds, batch_data
        )
        bpr_loss_hypergraph = cal_bpr_loss(anc_hypergraph_embeds, pos_hypergraph_embeds, neg_hypergraph_embeds) / anc_hypergraph_embeds.shape[0]
        # 总 BPR 损失
        bpr_loss = bpr_loss_lightgcn + bpr_loss_hypergraph
        # ========== 超图相关代码结束 ==========
        # 语义嵌入（用于知识蒸馏）
        usrprf_embeds = self.mlp(self.usrprf_embeds)
        itmprf_embeds = self.mlp(self.itmprf_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)
        # 正则化损失
        reg_loss = self.reg_weight * reg_params(self)
        # 知识蒸馏损失（InfoNCE）
        kd_loss = cal_infonce_loss(anc_embeds, ancprf_embeds, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos_embeds, posprf_embeds, posprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(neg_embeds, negprf_embeds, negprf_embeds, self.kd_temperature)
        kd_loss /= anc_embeds.shape[0]
        kd_loss *= self.kd_weight
        loss = bpr_loss + reg_loss + kd_loss
        losses = {
            'bpr_loss': bpr_loss,
            'bpr_loss_lightgcn': bpr_loss_lightgcn,
            'bpr_loss_hypergraph': bpr_loss_hypergraph,
            'reg_loss': reg_loss,
            'kd_loss': kd_loss
        }
        return loss, losses
    def full_predict(self, batch_data):
        """全排序预测"""
        # ========== 超图相关代码 ==========
        user_embeds, item_embeds, hypergraph_user_embeds, hypergraph_item_embeds = self.forward(self.adj, 1.0)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long
        # LightGCN 的预测分数
        pck_lightgcn_user_embeds = user_embeds[pck_users]
        lightgcn_preds = pck_lightgcn_user_embeds @ item_embeds.T
        # 超图的预测分数
        pck_hypergraph_user_embeds = hypergraph_user_embeds[pck_users]
        hypergraph_preds = pck_hypergraph_user_embeds @ hypergraph_item_embeds.T
        # 融合两种预测分数
        full_preds = lightgcn_preds + self.hypergraph_weight * hypergraph_preds
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
        # ========== 超图相关代码结束 ==========
