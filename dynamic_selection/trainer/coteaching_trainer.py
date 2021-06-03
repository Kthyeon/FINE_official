import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List
from torchvision.utils import make_grid
from base import BaseTrainer
from utils.util import inf_loop
import sys
from sklearn.mixture import GaussianMixture
import pdb
import numpy as np
import copy

class CoteachingTrainer(BaseTrainer):
    """
    DefaultTrainer class
    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, train_criterion, metrics, optimizer, config, data_loader, parse,
                 valid_data_loader=None, test_data_loader=None, teacher = None, lr_scheduler=None, len_epoch=None, val_criterion=None, mode=None, entropy=False, threshold=0.1,
                 epoch_decay_start=80, n_epoch=200, learning_rate=0.001):
        super().__init__(model, train_criterion, metrics, optimizer, config, val_criterion,parse)
        self.config = config
        self.data_loader = data_loader
        self.mode = mode
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        
        # Specific attribute for coteaching
        self.model_1, self.model_2 = model[0].to(self.device), model[1].to(self.device)
        self.optimizer_1, self.optimizer_2 = optimizer[0], optimizer[1]
        
        if lr_scheduler is not None:
            self.lr_scheduler_1, self.lr_scheduler_2 = lr_scheduler[0], lr_scheduler[1]
        else:
            self.lr_scheduler_1, self.lr_scheduler_2 = None, None

        # re-initialization model
        for m in self.model_1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
        for m in self.model_2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_test = self.test_data_loader is not None
#         self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_loss_list: List[float] = []
        self.val_loss_list: List[float] = []
        self.test_loss_list: List[float] = []
        #Visdom visualization
        
        self.entropy = entropy
        if self.entropy:
            self.entro_loss = Entropy(threshold)
            
        # Adjust learning rate and betas for Adam Optimizer
        
        self.epoch_decay_start = epoch_decay_start
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate
        
        mom1, mom2 = 0.9, 0.1
        self.alpha_plan = [self.learning_rate] * self.n_epoch
        self.beta1_plan = [mom1] * self.n_epoch

        for i in range(self.epoch_decay_start, self.n_epoch):
            self.alpha_plan[i] = float(self.n_epoch - i) / (self.n_epoch - self.epoch_decay_start) * self.learning_rate
            self.beta1_plan[i] = mom2

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr']=self.alpha_plan[epoch-1]
            param_group['betas']=(self.beta1_plan[epoch-1], 0.999)

    def _eval_metrics(self, output, label):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, label)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss_1, total_loss_2 = 0, 0
        total_metrics_1, total_metrics_2 = np.zeros(len(self.metrics)), np.zeros(len(self.metrics))
        total_metrics_gt_1, total_metrics_gt_2 = np.zeros(len(self.metrics)), np.zeros(len(self.metrics))
        total_instances_1, total_instances_2 = 0, 0
        clean_instances_1, clean_instances_2 = 0, 0

        with tqdm(self.data_loader) as progress:
            for batch_idx, (data, label, indexs, gt) in enumerate(progress):
                progress.set_description_str(f'Train epoch {epoch}')
                
                data, label = data.to(self.device), label.long().to(self.device)
                gt = gt.long().to(self.device)
                    
                _, output_1 = self.model_1(data)
                _, output_2 = self.model_2(data)

                loss_1, loss_2, clean_1, clean_2, total_1, total_2 = \
                self.train_criterion(output_1, output_2, label, gt, epoch, indexs, epoch*batch_idx)
                
                self.optimizer_1.zero_grad()
                loss_1.backward()
                self.optimizer_1.step()
                
                self.optimizer_2.zero_grad()
                loss_2.backward()
                self.optimizer_2.step()

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.writer.add_scalar('loss_1', loss_1.item())
                self.writer.add_scalar('loss_2', loss_2.item())
#                 self.train_loss_list.append(loss.item())
                
                total_loss_1 += loss_1.item()
                total_metrics_1 += self._eval_metrics(output_1, label)
                total_metrics_gt_1 += self._eval_metrics(output_1, gt)
                total_loss_2 += loss_2.item()
                total_metrics_2 += self._eval_metrics(output_2, label)
                total_metrics_gt_2 += self._eval_metrics(output_2, gt)
                
                clean_instances_1 += clean_1
                clean_instances_2 += clean_2
                total_instances_1 += total_1
                total_instances_2 += total_2

                if batch_idx == self.len_epoch:
                    break

        purity_1 = float(clean_instances_1)/float(total_instances_1)
        purity_2 = float(clean_instances_2)/float(total_instances_2)
        
        log = {
            'loss_1': total_loss_1 / self.len_epoch,
            'loss_2': total_loss_2 / self.len_epoch,
            'metrics_1': (total_metrics_1 / self.len_epoch).tolist(),
            'metrics_gt_1': (total_metrics_gt_1 / self.len_epoch).tolist(),
            'metrics_2': (total_metrics_2 / self.len_epoch).tolist(),
            'metrics_gt_2': (total_metrics_gt_2 / self.len_epoch).tolist(),
            'learning rate': self.lr_scheduler_1.get_lr() if self.lr_scheduler_1 is not None else self.alpha_plan[epoch-1],
            'purity_1': '{} = {}/{}'.format(purity_1,
                                            clean_instances_1, total_instances_1),
            'purity_2': '{} = {}/{}'.format(purity_2,
                                            clean_instances_2, total_instances_2)
        }


        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)
        if self.do_test:
            test_log, test_meta = self._test_epoch(epoch)
            log.update(test_log)
        else: 
            test_meta = [0,0]

        if self.lr_scheduler_1 is None:
            self.adjust_learning_rate(self.optimizer_1, epoch)
            self.adjust_learning_rate(self.optimizer_2, epoch)
        else:
            print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            print ('Ang')
            self.lr_scheduler_1.step()
            self.lr_scheduler_2.step()
        
#         if self.lr_scheduler_1 is not None:
#             self.lr_scheduler_1.step()
#         if self.lr_scheduler_2 is not None:
#             self.lr_scheduler_2.step()
            
        return log


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()

        total_val_loss_1, total_val_loss_2 = 0, 0
        total_val_metrics_1, total_val_metrics_2 = np.zeros(len(self.metrics)), np.zeros(len(self.metrics))
        with torch.no_grad():
            with tqdm(self.valid_data_loader) as progress:
                for batch_idx, (data, label, _, _) in enumerate(progress):
                    progress.set_description_str(f'Valid epoch {epoch}')
                    data, label = data.to(self.device), label.to(self.device)
#                     _, output = self.model(data)
                    _, output_1 = self.model_1(data)
                    _, output_2 = self.model_2(data)
                    
                    loss_1 = self.val_criterion()(output_1, label)
                    loss_2 = self.val_criterion()(output_2, label)

                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                    self.writer.add_scalar('loss_1', loss.item())
                    self.writer.add_scalar('loss_2', loss.item())
#                     self.val_loss_list.append(loss.item())

                    total_val_loss_1 += loss_1.item()
                    total_val_metrics_1 += self._eval_metrics(output_1, label)
                    total_val_loss_2 += loss_2.item()
                    total_val_metrics_2 += self._eval_metrics(output_2, label)
        
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss_1': total_val_loss_1 / len(self.valid_data_loader),
            'val_metrics_1': (total_val_metrics_1 / len(self.valid_data_loader)).tolist(),
            'val_loss_2': total_val_loss_2 / len(self.valid_data_loader),
            'val_metrics_2': (total_val_metrics_2 / len(self.valid_data_loader)).tolist()
        }

    def _test_epoch(self, epoch):
        """
        Test after training an epoch
        :return: A log that contains information about test
        Note:
            The Test metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_test_loss_1, total_test_loss_2 = 0, 0
        total_test_metrics_1, total_test_metrics_2 = np.zeros(len(self.metrics)), np.zeros(len(self.metrics))
        results = np.zeros((len(self.test_data_loader.dataset), self.config['num_classes']), dtype=np.float32)
        tar_ = np.zeros((len(self.test_data_loader.dataset),), dtype=np.float32)
        with torch.no_grad():
            with tqdm(self.test_data_loader) as progress:
                for batch_idx, (data, label,indexs,_) in enumerate(progress):
                    progress.set_description_str(f'Test epoch {epoch}')
                    data, label = data.to(self.device), label.to(self.device)
#                     _, output = self.model(data)
                    
                    _, output_1 = self.model_1(data)
                    _, output_2 = self.model_2(data)
                    loss_1 = self.val_criterion()(output_1, label)
                    loss_2 = self.val_criterion()(output_2, label)    
                    
                    self.writer.set_step((epoch - 1) * len(self.test_data_loader) + batch_idx, 'test')

                    total_test_loss_1 += loss_1.item()
                    total_test_metrics_1 += self._eval_metrics(output_1, label)
                    total_test_loss_2 += loss_2.item()
                    total_test_metrics_2 += self._eval_metrics(output_2, label)
                    
                    if total_test_metrics_1[0] > total_test_metrics_2[0]:
                        total_test_metrics = total_test_metrics_1
                    else:
                        total_test_metrics = total_test_metrics_2
                    
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                    results[indexs.cpu().detach().numpy().tolist()] = output_1.cpu().detach().numpy().tolist()
                    tar_[indexs.cpu().detach().numpy().tolist()] = label.cpu().detach().numpy().tolist()

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'test_loss_1': total_test_loss_1 / len(self.test_data_loader),
            'test_metrics_1': (total_test_metrics_1 / len(self.test_data_loader)).tolist(),
            'test_loss_2': total_test_loss_2 / len(self.test_data_loader),
            'test_metrics_2': (total_test_metrics_2 / len(self.test_data_loader)).tolist(),
            'test_metrics': (total_test_metrics / len(self.test_data_loader)).tolist()
            
        },[results,tar_]

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)