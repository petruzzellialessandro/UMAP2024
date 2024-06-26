from .loggers import *
from .utils import AverageMeterSet

STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import json
from abc import *
from pathlib import Path


class AbstractTrainer(metaclass=ABCMeta):
    def __init__(self, args_dict, model, train_loader, val_loader, test_loader, export_root):
        self.args_dict = args_dict
        self.device = args_dict['device']
        self.model = model.to(self.device)
        self.is_parallel = args_dict['num_gpu'] > 1
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = self._create_optimizer()
        if args_dict['enable_lr_schedule']:
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args_dict['decay_step'], gamma=args_dict['gamma'])

        self.num_epochs = args_dict['num_epochs']
        self.metric_ks = args_dict['metric_ks']
        self.best_metric = args_dict['best_metric']
        self.embedding_conf = args_dict['embedding_conf']
        self.dim = str(args_dict['dim'])
        self.dataset = args_dict['dataset']

        self.export_root = export_root
        self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
        self.add_extra_loggers()
        self.logger_service = LoggerService(self.train_loggers, self.val_loggers)
        self.log_period_as_iter = args_dict['log_period_as_iter']

    @abstractmethod
    def add_extra_loggers(self):
        pass

    @abstractmethod
    def log_extra_train_info(self, log_data):
        pass

    @abstractmethod
    def log_extra_val_info(self, log_data):
        pass

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    @abstractmethod
    def calculate_metrics(self, batch):
        pass

    def train(self):
        accum_iter = 0
        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            value, model = self.validate(epoch, accum_iter)
        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()

    def train_one_epoch(self, epoch, accum_iter):
        self.model.train()

        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch_size = batch[0].size(0)
            batch = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            loss.backward()
            #nn.utils.clip_grad_norm_(self.model.parameters(), 0.5, norm_type=2)
            self.optimizer.step()
            average_meter_set.update('loss', loss.item())
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.3f} '.format(epoch+1, average_meter_set['loss'].avg))

            accum_iter += batch_size
        if self.args_dict['enable_lr_schedule']:
            self.lr_scheduler.step()

            if self._needs_to_log(accum_iter):
                tqdm_dataloader.set_description('Logging to Tensorboard')
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch+1,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.log_extra_train_info(log_data)
                self.logger_service.log_train(log_data)

        return accum_iter

    def validate(self, epoch, accum_iter):
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]]
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
            self.log_extra_val_info(log_data)
            self.logger_service.log_val(log_data)
            metric_avg = (10*average_meter_set['Recall@1'].avg +
            3*average_meter_set['Recall@10'].avg +
            average_meter_set['Recall@50'].avg)
            print("Metric AVG:  {:.3f}".format(metric_avg))
        return metric_avg, self.model

    def test(self):
        print('Test best model with test set!')

        best_model = torch.load(os.path.join(self.export_root, 'trained_models', f'best_acc_model-{self.dataset}_{self.embedding_conf}={self.dim}.pth')).get('model_state_dict')
        self.model.load_state_dict(best_model)
        self.model.eval()

        average_meter_set = AverageMeterSet()
        results_stats = []

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch)
                results_stats.extend(self.get_stast_infos(batch))

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]]
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            average_metrics = average_meter_set.averages()
            with open(os.path.join(self.export_root, 'logs', f'test_metrics_{self.dataset}_{self.embedding_conf}={self.dim}.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
            results_stats_dicts = {i:pos for i, pos in enumerate(results_stats)}
            if not os.path.exists(os.path.join(self.export_root, 'stats')):
                os.mkdir(os.path.join(self.export_root, 'stats'))
            with open(os.path.join(self.export_root, 'stats', f'stats_{self.dataset}_{self.embedding_conf}={self.dim}.json'), 'w') as f:
                json.dump(results_stats_dicts, f, indent=4)
            print(average_metrics)

    def _create_optimizer(self):
        args_dict = self.args_dict
        if args_dict['optimizer'].lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=args_dict['lr'], weight_decay=args_dict['weight_decay'])
        elif args_dict['optimizer'].lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=args_dict['lr'], weight_decay=args_dict['weight_decay'], momentum=args_dict['momentum'])
        else:
            raise ValueError

    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('trained_models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss', graph_name='Loss', group_name='Train'),
        ]

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
        val_loggers.append(RecentModelLogger(model_checkpoint, filename=f'best_acc_model-{self.dataset}_{self.embedding_conf}={self.dim}.pth'))
        val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric, filename=f'best_acc_model-{self.dataset}_{self.embedding_conf}={self.dim}.pth'))
        return writer, train_loggers, val_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args_dict['train_batch_size'] and accum_iter != 0