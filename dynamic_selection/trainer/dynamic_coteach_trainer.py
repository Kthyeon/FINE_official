import os
# os.chdir('../')
# print (os.getcwd())

from selection.svd_classifier import *
from selection.gmm import *
from selection.util import *
import data_loader.data_loaders as module_data

import copy
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

class FCoteachingTrainer(BaseTrainer):
    """
    DefaultTrainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, train_criterion, metrics, optimizer, config, data_loader, parse,
                 valid_data_loader=None, test_data_loader=None, teacher = None, lr_scheduler=None, len_epoch=None, val_criterion=None, mode=None, entropy=False, threshold = 0.1):
        super().__init__(model, train_criterion, metrics, optimizer, config, val_criterion, parse)
        self.config = config
        self.data_loader = data_loader
        self.mode = mode
        self.parse = parse
        
        #####################################
        # Specific attribute for coteaching #
        #####################################
        
        self.model_1, self.model_2 = copy.deepcopy(model).to(self.device), copy.deepcopy(model).to(self.device)
        trainable_params1 = filter(lambda p: p.requires_grad, self.model_1.parameters())
        trainable_params2 = filter(lambda p: p.requires_grad, self.model_2.parameters())
        self.optimizer_1 = config.initialize('optimizer', torch.optim, [{'params': trainable_params1}])
        self.optimizer_2 = config.initialize('optimizer', torch.optim, [{'params': trainable_params2}])
        self.lr_scheduler_1 = config.initialize('lr_scheduler', torch.optim.lr_scheduler, self.optimizer_1)
        self.lr_scheduler_2 = config.initialize('lr_scheduler', torch.optim.lr_scheduler, self.optimizer_2)
        
        # CODE FOR RUNNING. REDUNDANT
        self.optimizer = self.optimizer_1
        self.lr_scheduler = self.lr_scheduler_1
            
        # re-initialization model
        for m in self.model_1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
        for m in self.model_2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
        ####################################
        ############# COMPLETE #############
        ####################################
        
        self.warm_up = parse.warmup
        self.every = parse.every
        
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
            self.len_epoch_1 = self.len_epoch_2 = self.len_epoch
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
            self.len_epoch_1 = self.len_epoch_2 = self.len_epoch
        self.dynamic_train_data_loader_1 = copy.deepcopy(data_loader)
        self.dynamic_train_data_loader_2 = copy.deepcopy(data_loader)
        self.valid_data_loader = valid_data_loader
        
        self.orig_data_loader = getattr(module_data, self.config['data_loader']['type'])(
            self.config['data_loader']['args']['data_dir'],
            batch_size=self.config['data_loader']['args']['batch_size'],
            shuffle=False,
            validation_split=0.1,
            num_batches=self.config['data_loader']['args']['num_batches'],
            training=True,
            num_workers=self.config['data_loader']['args']['num_workers'],
            pin_memory=self.config['data_loader']['args']['pin_memory'])
        
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_test = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_loss_list: List[float] = []
        self.val_loss_list: List[float] = []
        self.test_loss_list: List[float] = []
        self.purity_1 = self.purity_2 = (data_loader.train_dataset.train_labels == \
                       data_loader.train_dataset.train_labels_gt).sum() / len(data_loader.train_dataset)
        self.teacher_idx_1, self.teacher_idx_2 = None, None
        #Visdom visualization
        
        self.entropy = entropy
        if self.entropy: self.entro_loss = Entropy(threshold)
            
            
    def update_dataloader(self, epoch):
        
        with torch.no_grad():
            current_features_1, current_labels_1 = get_features(self.model_1, self.orig_data_loader)
            current_features_2, current_labels_2 = get_features(self.model_2, self.orig_data_loader)
            datanum = len(current_labels_1)
            
            if self.teacher_idx_1 is not None:
                prev_features_1, prev_labels_1 = current_features_1[self.teacher_idx_1], current_labels_1[self.teacher_idx_1]
                prev_features_2, prev_labels_2 = current_features_2[self.teacher_idx_2], current_labels_2[self.teacher_idx_2]
            else:
                prev_features_1, prev_labels_1 = current_features_1, current_labels_1
                prev_features_2, prev_labels_2 = current_features_2, current_labels_2
                
            self.teacher_idx_1 = fine(current_features_2, current_labels_2, fit=self.parse.distill_mode, prev_features=prev_features_2, prev_labels=prev_labels_2)
            self.teacher_idx_2 = fine(current_features_1, current_labels_1, fit=self.parse.distill_mode, prev_features=prev_features_1, prev_labels=prev_labels_1)
            
        curr_data_loader_1 = getattr(module_data, self.config['data_loader']['type'])(
            self.config['data_loader']['args']['data_dir'],
            batch_size=self.config['data_loader']['args']['batch_size'],
            shuffle=self.config['data_loader']['args']['shuffle'],
            validation_split=0.1,
            num_batches=self.config['data_loader']['args']['num_batches'],
            training=True,
            num_workers=self.config['data_loader']['args']['num_workers'],
            pin_memory=self.config['data_loader']['args']['pin_memory'],
            teacher_idx=self.teacher_idx_1)
        
        curr_data_loader_2 = getattr(module_data, self.config['data_loader']['type'])(
            self.config['data_loader']['args']['data_dir'],
            batch_size=self.config['data_loader']['args']['batch_size'],
            shuffle=self.config['data_loader']['args']['shuffle'],
            validation_split=0.1,
            num_batches=self.config['data_loader']['args']['num_batches'],
            training=True,
            num_workers=self.config['data_loader']['args']['num_workers'],
            pin_memory=self.config['data_loader']['args']['pin_memory'],
            teacher_idx=self.teacher_idx_2)
        
        self.selected, self.precision, self.recall, self.f1, self.specificity, self.accuracy = return_statistics(self.orig_data_loader, self.teacher_idx_1)
        self.selected, self.precision, self.recall, self.f1, self.specificity, self.accuracy = return_statistics(self.orig_data_loader, self.teacher_idx_2)
        
        return curr_data_loader_1, curr_data_loader_2
        

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
        if epoch % self.every == 3 and epoch > self.warm_up: # 
            self.dynamic_train_data_loader_1, self.dynamic_train_data_loader_2 = self.update_dataloader(epoch)
            self.len_epoch_1 = len(self.dynamic_train_data_loader_1)
            self.len_epoch_2 = len(self.dynamic_train_data_loader_2)
            self.purity_1 = (self.dynamic_train_data_loader_1.train_dataset.train_labels == \
                             self.dynamic_train_data_loader_1.train_dataset.train_labels_gt).sum() / \
                            len(self.dynamic_train_data_loader_1.train_dataset)
            self.purity_2 = (self.dynamic_train_data_loader_2.train_dataset.train_labels == \
                             self.dynamic_train_data_loader_2.train_dataset.train_labels_gt).sum() / \
                            len(self.dynamic_train_data_loader_2.train_dataset)
            
#         if epoch > 30:
#             self.train_criterion = CCELoss()
        
        self.model.train()

        total_loss_1, total_loss_2 = 0, 0
        total_metrics_1, total_metrics_2 = np.zeros(len(self.metrics)), np.zeros(len(self.metrics))
        total_metrics_gt_1, total_metrics_gt_2 = np.zeros(len(self.metrics)), np.zeros(len(self.metrics))

        with tqdm(self.dynamic_train_data_loader_1) as progress:
            for batch_idx, (data, label, indexs, gt) in enumerate(progress):
                progress.set_description_str(f'Train epoch {epoch}')
                
                data, label = data.to(self.device), label.long().to(self.device)
                gt = gt.long().to(self.device)
                
                _, output = self.model_1(data)
                loss_1 = self.train_criterion(output, label, indexs.cpu().detach().numpy().tolist())

                self.optimizer_1.zero_grad()
                loss_1.backward()
                self.optimizer_1.step()

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.writer.add_scalar('loss_1', loss_1.item())
                self.train_loss_list.append(loss_1.item())
                total_loss_1 += loss_1.item()
                total_metrics_1 += self._eval_metrics(output, label)
                total_metrics_gt_1 += self._eval_metrics(output, gt)

                if batch_idx % self.log_step == 0:
                    progress.set_postfix_str(' {} Loss: {:.6f}'.format(
                        self._progress(batch_idx),
                        loss_1.item()))
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                if batch_idx == self.len_epoch_1:
                    break
                    
        with tqdm(self.dynamic_train_data_loader_2) as progress:
            for batch_idx, (data, label, indexs, gt) in enumerate(progress):
                progress.set_description_str(f'Train epoch {epoch}')
                
                data, label = data.to(self.device), label.long().to(self.device)
                gt = gt.long().to(self.device)
                
                _, output = self.model_2(data)
                loss_2 = self.train_criterion(output, label, indexs.cpu().detach().numpy().tolist())

                self.optimizer_2.zero_grad()
                loss_2.backward()
                self.optimizer_2.step()

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.writer.add_scalar('loss_2', loss_2.item())
                self.train_loss_list.append(loss_2.item())
                total_loss_2 += loss_2.item()
                total_metrics_2 += self._eval_metrics(output, label)
                total_metrics_gt_2 += self._eval_metrics(output, gt)
                
                if batch_idx % self.log_step == 0:
                    progress.set_postfix_str(' {} Loss: {:.6f}'.format(
                        self._progress(batch_idx),
                        loss_2.item()))
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                if batch_idx == self.len_epoch_2:
                    break

        log = {
            'loss_1': total_loss_1 / self.len_epoch,
            'loss_2': total_loss_2 / self.len_epoch,
            'metrics_1': (total_metrics_1 / self.len_epoch).tolist(),
            'metrics_gt_1': (total_metrics_gt_1 / self.len_epoch).tolist(),
            'metrics_2': (total_metrics_2 / self.len_epoch).tolist(),
            'metrics_gt_2': (total_metrics_gt_2 / self.len_epoch).tolist(),
            'learning rate': self.lr_scheduler_1.get_last_lr(),
            'purity_1:': '{} = {}/{}'.format(self.purity_1, (self.dynamic_train_data_loader_1.train_dataset.train_labels == \
                   self.dynamic_train_data_loader_1.train_dataset.train_labels_gt).sum(), len(self.dynamic_train_data_loader_1.train_dataset)),
            'purity_2:': '{} = {}/{}'.format(self.purity_2, (self.dynamic_train_data_loader_2.train_dataset.train_labels == \
                   self.dynamic_train_data_loader_2.train_dataset.train_labels_gt).sum(), len(self.dynamic_train_data_loader_2.train_dataset))
        }


        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)
        if self.do_test:
            test_log, test_meta = self._test_epoch(epoch)
            log.update(test_log)
        else: 
            test_meta = [0,0]


        if self.lr_scheduler is not None:
            self.lr_scheduler_1.step()
            self.lr_scheduler_2.step()
            
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
                    self.writer.add_scalar('loss_1', loss_1.item())
                    self.writer.add_scalar('loss_2', loss_2.item())
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


    def _warmup_epoch(self, epoch):
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        self.model.train()

        data_loader = self.data_loader#self.loader.run('warmup')


        with tqdm(data_loader) as progress:
            for batch_idx, (data, label, _, indexs , _) in enumerate(progress):
                progress.set_description_str(f'Warm up epoch {epoch}')

                data, label = data.to(self.device), label.long().to(self.device)

                self.optimizer.zero_grad()
                _, output = self.model(data)
                out_prob = torch.nn.functional.softmax(output).data.detach()

                self.train_criterion.update_hist(indexs.cpu().detach().numpy().tolist(), out_prob)

                loss = torch.nn.functional.cross_entropy(output, label)

                loss.backward() 
                self.optimizer.step()

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.writer.add_scalar('loss', loss.item())
                self.train_loss_list.append(loss.item())
                total_loss += loss.item()
                total_metrics += self._eval_metrics(output, label)


                if batch_idx % self.log_step == 0:
                    progress.set_postfix_str(' {} Loss: {:.6f}'.format(
                        self._progress(batch_idx),
                        loss.item()))
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                if batch_idx == self.len_epoch:
                    break
        if hasattr(self.data_loader, 'run'):
            self.data_loader.run()
        log = {
            'loss': total_loss / self.len_epoch,
            'noise detection rate' : 0.0,
            'metrics': (total_metrics / self.len_epoch).tolist(),
            'learning rate': self.lr_scheduler.get_last_lr()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)
        if self.do_test:
            test_log, test_meta = self._test_epoch(epoch)
            log.update(test_log)
        else: 
            test_meta = [0,0]

        return log


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)