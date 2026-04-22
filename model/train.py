
import os
import torch
import torch.optim as optim
from copy import deepcopy
from tqdm import tqdm
import argparse
from model import LightGCN_hypergraph
from data_loader import DataHandler
from metrics import Metric
from logger import Logger
from utils import init_seed
from configurator import load_config
class Trainer(object):
    def __init__(self, data_handler, logger, config):
        self.data_handler = data_handler
        self.logger = logger
        self.config = config
        self.metric = Metric(config['test']['metrics'], config['test']['k'])
    def create_optimizer(self, model):
        optim_config = self.config['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(
                model.parameters,
                lr=optim_config['lr'],
                weight_decay=optim_config['weight_decay']
            )
    def train_epoch(self, model, epoch_idx):
        # prepare training data
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs
        # for recording loss
        loss_log_dict = {}
        ep_loss = 0
        steps = len(train_dataloader.dataset) // self.config['train']['batch_size']
        # start this epoch
        model.train
        for _, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            self.optimizer.zero_grad
            batch_data = list(map(lambda x: x.long.to(self.config['device']), tem))
            loss, loss_dict = model.cal_loss(batch_data)
            ep_loss += loss.item
            loss.backward
            self.optimizer.step
            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val
        # log loss
        if 'log_loss' in self.config['train'] and self.config['train']['log_loss']:
            self.logger.log(loss_log_dict, save_to_log=False, print_to_console=True)
    def train(self, model):
        now_patience = 0
        best_epoch = 0
        best_recall = -1e9
        self.create_optimizer(model)
        train_config = self.config['train']
        for epoch_idx in range(train_config['epoch']):
            # train
            self.train_epoch(model, epoch_idx)
            # evaluate
            if epoch_idx % train_config['test_step'] == 0:
                eval_result = self.evaluate(model, epoch_idx)
                if eval_result['recall'][-1] > best_recall:
                    now_patience = 0
                    best_epoch = epoch_idx
                    best_recall = eval_result['recall'][-1]
                    best_state_dict = deepcopy(model.state_dict)
                else:
                    now_patience += 1
                # early stop
                if now_patience == train_config['patience']:
                    break
        # evaluation again
        model = LightGCN_hypergraph(self.config, self.data_handler).to(self.config['device'])
        model.load_state_dict(best_state_dict)
        self.evaluate(model)
        # final test
        model = LightGCN_hypergraph(self.config, self.data_handler).to(self.config['device'])
        model.load_state_dict(best_state_dict)
        test_result = self.test(model)
        # save result
        self.save_model(model)
        self.logger.log("Best Epoch {}. Final test result: {}.".format(best_epoch, test_result))
    def evaluate(self, model, epoch_idx=None):
        """评估模型"""
        model.eval
        eval_result = self.metric.eval(model, self.data_handler.valid_dataloader, self.config['device'])
        self.logger.log_eval(eval_result, self.config['test']['k'], data_type='Validation set', epoch_idx=epoch_idx)
        return eval_result
    def test(self, model):
        model.eval
        eval_result = self.metric.eval(model, self.data_handler.test_dataloader, self.config['device'])
        self.logger.log_eval(eval_result, self.config['test']['k'], data_type='Test set')
        return eval_result
    def save_model(self, model):
        if self.config['train']['save_model']:
            model_state_dict = model.state_dict
            model_name = self.config['model']['name']
            save_dir_path = './checkpoint/{}'.format(model_name)
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)
            torch.save(
                model_state_dict,
                '{}/{}-{}-{}.pth'.format(
                    save_dir_path,
                    model_name,
                    self.config['data']['name'],
                    self.config['train']['seed']
                )
            )
            self.logger.log(
                "Save model parameters to {}".format(
                    '{}/{}-{}-{}.pth'.format(
                        save_dir_path,
                        model_name,
                        self.config['data']['name'],
                        self.config['train']['seed']
                    )
                )
            )
def train(config_path='config.yml'):
    # First Step: 加载配置
    config = load_config(config_path)
    # Second Step: 初始化随机种子
    if config['train']['reproducible']:
        init_seed(config['train']['seed'])
    # Third Step: 创建数据处理器
    data_handler = DataHandler(config)
    data_handler.load_data
    # Fourth Step: 创建模型（直接创建，不使用build_model工厂模式）
    model = LightGCN_hypergraph(config, data_handler).to(config['device'])
    # Fifth Step: 创建logger
    logger = Logger(config)
    # Sixth Step: 创建trainer
    trainer = Trainer(data_handler, logger, config)
    # 打印超图权重
    logger.log("超图权重: {}".format(model.hypergraph_weight))
    logger.log("=" * 60)
    # Seventh Step: 开始训练
    trainer.train(model)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LightGCN Hypergraph 独立训练脚本')
    parser.add_argument('--config', type=str, default='config.yml', help='配置文件路径')
    args = parser.parse_args
    train(args.config)
