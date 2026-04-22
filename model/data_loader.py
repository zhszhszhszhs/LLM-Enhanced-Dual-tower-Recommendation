"""独立的数据加载器"""
import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sp
import torch as t
import torch.utils.data as data
# 全局配置
_global_data_config = {}
class PairwiseTrnData(data.Dataset):
    """成对训练数据"""
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok
        self.negs = np.zeros(len(self.rows)).astype(np.int32)
    def sample_negs(self):
        """采样负样本"""
        # 从全局配置获取item_num
        item_num = _global_data_config.get('item_num', None)
        if item_num is None:
            # 如果全局配置未设置，从coomat推断（fallback）
            item_num = max(self.cols) + 1 if len(self.cols) > 0 else 0
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(item_num)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]
class AllRankTstData(data.Dataset):
    """全排序测试数据"""
    def __init__(self, coomat, trn_mat):
        self.csrmat = (trn_mat.tocsr != 0) * 1.0
        user_pos_lists = [list for i in range(coomat.shape[0])]
        test_users = set
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            user_pos_lists[row].append(col)
            test_users.add(row)
        self.test_users = np.array(list(test_users))
        self.user_pos_lists = user_pos_lists
    def __len__(self):
        return len(self.test_users)
    def __getitem__(self, idx):
        pck_user = self.test_users[idx]
        pck_mask = self.csrmat[pck_user].toarray
        pck_mask = np.reshape(pck_mask, [-1])
        return pck_user, pck_mask
class DataHandler:
    """数据处理器"""
    def __init__(self, config):
        self.config = config
        dataset_name = config['data']['name']
        data_dir = config['data']['data_dir']
        if dataset_name == 'amazon':
            predir = f'{data_dir}/amazon/'
        elif dataset_name == 'yelp':
            predir = f'{data_dir}/yelp/'
        elif dataset_name == 'steam':
            predir = f'{data_dir}/steam/'
        elif dataset_name == 'el':
            predir = f'{data_dir}/el/'
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not supported")
        self.trn_file = predir + 'trn_mat.pkl'
        self.val_file = predir + 'val_mat.pkl'
        self.tst_file = predir + 'tst_mat.pkl'
    def _load_one_mat(self, file):
        """加载一个邻接矩阵"""
        with open(file, 'rb') as fs:
            mat = (pickle.load(fs) != 0).astype(np.float32)
        if type(mat) != coo_matrix:
            mat = coo_matrix(mat)
        return mat
    def _normalize_adj(self, mat):
        """拉普拉斯归一化"""
        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        return mat.dot(d_inv_sqrt_mat).transpose.dot(d_inv_sqrt_mat).tocoo
    def _make_torch_adj(self, mat, self_loop=False):
        """构建双向邻接矩阵"""
        # 使用config中的user_num和item_num
        user_num = self.config['data']['user_num']
        item_num = self.config['data']['item_num']
        if not self_loop:
            a = csr_matrix((user_num, user_num))
            b = csr_matrix((item_num, item_num))
        else:
            data = np.ones(user_num)
            row_indices = np.arange(user_num)
            column_indices = np.arange(user_num)
            a = csr_matrix((data, (row_indices, column_indices)), shape=(user_num, user_num))
            data = np.ones(item_num)
            row_indices = np.arange(item_num)
            column_indices = np.arange(item_num)
            b = csr_matrix((data, (row_indices, column_indices)), shape=(item_num, item_num))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose, b])])
        mat = (mat != 0) * 1.0
        mat = self._normalize_adj(mat)
        # 转换为 torch sparse tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(self.config['device'])
    def load_data(self):
        """加载所有数据"""
        trn_mat = self._load_one_mat(self.trn_file)
        val_mat = self._load_one_mat(self.val_file)
        tst_mat = self._load_one_mat(self.tst_file)
        self.trn_mat = trn_mat
        # 设置config中的user_num和item_num
        self.config['data']['user_num'], self.config['data']['item_num'] = trn_mat.shape
        # 设置全局配置
        global _global_data_config
        _global_data_config['item_num'] = self.config['data']['item_num']
        _global_data_config['user_num'] = self.config['data']['user_num']
        # 构建torch邻接矩阵
        self.torch_adj = self._make_torch_adj(trn_mat)
        # 处理gccf模型
        if self.config['model']['name'] == 'gccf':
            self.torch_adj = self._make_torch_adj(trn_mat, self_loop=True)
        # 创建数据加载器
        if self.config['train']['loss'] == 'pairwise':
            trn_data = PairwiseTrnData(trn_mat)
        elif self.config['train']['loss'] == 'pairwise_with_epoch_flag':
            # 注意：tmp版本可能没有PairwiseWEpochFlagTrnData，这里简化处理
            trn_data = PairwiseTrnData(trn_mat)
        val_data = AllRankTstData(val_mat, trn_mat)
        tst_data = AllRankTstData(tst_mat, trn_mat)
        self.test_dataloader = data.DataLoader(tst_data, batch_size=self.config['test']['batch_size'], shuffle=False, num_workers=0)
        self.valid_dataloader = data.DataLoader(val_data, batch_size=self.config['test']['batch_size'], shuffle=False, num_workers=0)
        self.train_dataloader = data.DataLoader(trn_data, batch_size=self.config['train']['batch_size'], shuffle=True, num_workers=0)
