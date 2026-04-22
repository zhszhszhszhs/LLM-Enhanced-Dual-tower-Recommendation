"""工具函数文件"""
import torch as t
import torch.nn as nn
import numpy as np
from numpy import random
import random as python_random
import os
def init_seed(seed=2023):
    """初始化随机种子"""
    python_random.seed(seed)
    random.seed(seed)  # numpy.random
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.benchmark = False
    t.backends.cudnn.deterministic = True
def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
    """计算BPR损失"""
    import torch.nn.functional as F
    pos_preds = (anc_embeds * pos_embeds).sum(-1)
    neg_preds = (anc_embeds * neg_embeds).sum(-1)
    return t.sum(F.softplus(neg_preds - pos_preds))
def cal_infonce_loss(embeds1, embeds2, all_embeds2, temp=1.0):
    """计算InfoNCE损失（用于知识蒸馏）"""
    normed_embeds1 = embeds1 / t.sqrt(1e-8 + embeds1.square.sum(-1, keepdim=True))
    normed_embeds2 = embeds2 / t.sqrt(1e-8 + embeds2.square.sum(-1, keepdim=True))
    normed_all_embeds2 = all_embeds2 / t.sqrt(1e-8 + all_embeds2.square.sum(-1, keepdim=True))
    nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
    deno_term = t.log(t.sum(t.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
    cl_loss = (nume_term + deno_term).sum
    return cl_loss
def reg_params(model):
    """计算L2正则化损失"""
    reg_loss = 0
    for W in model.parameters:
        reg_loss += W.norm(2).square
    return reg_loss
class SpAdjEdgeDrop(nn.Module):
    """稀疏邻接矩阵边dropout"""
    def __init__(self):
        super(SpAdjEdgeDrop, self).__init__
    def forward(self, adj, keep_rate):
        if keep_rate == 1.0:
            return adj
        vals = adj._values
        idxs = adj._indices
        edgeNum = vals.size
        mask = (t.rand(edgeNum) + keep_rate).floor.type(t.bool)
        newVals = vals[mask]
        newIdxs = idxs[:, mask]
        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)
