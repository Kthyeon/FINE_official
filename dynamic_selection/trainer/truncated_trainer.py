import numpy as np
import torch
from tqdm import tqdm
from .default_trainer import DefaultTrainer



class TruncatedTrainer(DefaultTrainer):
    def __init__(self, model, train_criterion, metrics, optimizer, config, data_loader, parse,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, len_epoch=None, val_criterion=None, teacher = None, mode = None,entropy = False,threshold = 0.1):
        super().__init__(model, train_criterion, metrics, optimizer, config, data_loader,parse,
                         valid_data_loader=valid_data_loader,
                         test_data_loader=test_data_loader,
                         lr_scheduler=lr_scheduler,
                         len_epoch=len_epoch,
                         val_criterion=val_criterion,
                         teacher = teacher,
                         mode = mode,
                         entropy = entropy,
                         threshold = threshold)
        
        self.start_prune = 40
        
#     def _eval_metrics(self, output, label):
#         return super()._eval_metrics(output, label)
    
    def _train_epoch(self, epoch):
        
        if epoch > self.start_prune and (epoch + 1) % 10 == 0:
            self.model.eval()
            with tqdm(self.data_loader) as progress:
                for batch_idx, (data, label, indexs, _) in enumerate(progress):
                    data, label = data.to(self.device), label.long().to(self.device)
                    model_represent, output = self.model(data)
                    
                    self.train_criterion.update_weight(output, label, indexs.cpu().detach().numpy().tolist())
                
        return super()._train_epoch(epoch)
    
    def _warmup_epoch(self, epoch):
        pass