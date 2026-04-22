"""日志记录器"""
import os
import logging
import datetime


def get_local_time():
    return datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')


class Logger(object):
    def __init__(self, config, log_configs=True):
        model_name = config['model']['name']
        log_dir_path = './log/{}'.format(model_name)
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        self.logger = logging.getLogger('train_logger')
        self.logger.setLevel(logging.INFO)
        dataset_name = config['data']['name']
        # 简化版本：不使用tune功能
        log_file = logging.FileHandler(
            '{}/{}_{}.log'.format(log_dir_path, dataset_name, get_local_time()),
            'a',
            encoding='utf-8'
        )
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        log_file.setFormatter(formatter)
        self.logger.addHandler(log_file)
        if log_configs:
            tmp_configs = {}
            tmp_configs['optimizer'] = config['optimizer']
            tmp_configs['train'] = config['train']
            tmp_configs['test'] = config['test']
            tmp_configs['data'] = config['data']
            tmp_configs['model'] = config['model']
            self.log(tmp_configs)
    def log(self, message, save_to_log=True, print_to_console=True):
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)
    def log_loss(self, epoch_idx, loss_log_dict, save_to_log=True, print_to_console=True):
        # 需要epoch总数，但这里简化处理（原框架从configs获取）
        message = '[Epoch {:3d}] '.format(epoch_idx)
        for loss_name in loss_log_dict:
            message += '{}: {:.4f} '.format(loss_name, loss_log_dict[loss_name])
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)
    def log_eval(self, eval_result, k, data_type, save_to_log=True, print_to_console=True, epoch_idx=None):
        if epoch_idx is not None:
            message = 'Epoch {:3d} {} '.format(epoch_idx, data_type)
        else:
            message = '{} '.format(data_type)
        for metric in eval_result:
            message += '['
            for i in range(len(k)):
                message += '{}@{}: {:.4f} '.format(metric, k[i], eval_result[metric][i])
            message += '] '
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)
