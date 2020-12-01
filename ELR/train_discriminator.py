import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import torch.nn.functional as F
import sys
import json
import pandas as pd
from tqdm import tqdm
import time
import collections

import data_loader.data_loaders as module_data
import loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.conv1_1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(10, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1, 4, 1, 0)

    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.sigmoid(self.conv4(x))
        return x

def get_clean_data(config):
    
    config_file = './hyperparams/multistep/config_cifar10_gce.json'
    with open(config_file, 'r') as f:
        config = json.load(f)

    resume_path = './saved/models/cifar10_resnet34_multistep_asym_40/1112_111900/model_best.pth'
    base_model = getattr(module_arch, config["arch"]['type'])()
    checkpoint = torch.load(resume_path)
    state_dict = checkpoint['state_dict']
    base_model.load_state_dict(state_dict)
    
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size= 1,
        shuffle=config['data_loader']['args']['shuffle'],
        validation_split=0.0,
        num_batches=config['data_loader']['args']['num_batches'],
        training=True,
        num_workers=config['data_loader']['args']['num_workers'],
        pin_memory=config['data_loader']['args']['pin_memory'],
        config=config
    )
    
    criterion = getattr(module_loss, 'GCELoss')(q=config['train_loss']['args']['q'],
                                                     k=config['train_loss']['args']['k'],
                                                     truncated=False)
    device = torch.device('cuda:1')
    base_model.eval()
    base_model.to(device)
    
    print("### Start: get clean data ###")
    index_list = np.empty((0,))
    loss_list = np.empty((0,))
    label_list = np.empty((0,))
    isClean_list = np.empty((0,))
    with tqdm(data_loader) as progress:
        for batch_idx, (data, label, index, label_gt) in enumerate(progress):
#             if batch_idx == 2000:
#                 break
            isClean = label == label_gt
            data = data.to(device)
            label = label.long().to(device)
            output = base_model(data)
            loss = criterion(output, label, None) # set index as None: not truncated
#             loss, pred = torch.max(F.softmax(output, dim=1), dim=1)
            loss = np.expand_dims(loss.detach().cpu().numpy(), axis=0)
#             loss = loss.detach().cpu().numpy()
            loss_list = np.concatenate((loss_list, loss), axis=0)
            index_list = np.concatenate((index_list, index))
            label_list = np.concatenate((label_list, label.cpu().numpy()))
            isClean_list = np.concatenate((isClean_list, isClean))
    df = pd.DataFrame({"loss" : loss_list,"index": index_list, "label": label_list, "isClean": isClean_list})
    threshold = 0.5
    clean_df = df[df.loss < threshold]
    
    print("### Clean dataset composition ###")
    print("Total : %d" % (len(clean_df)))
    for i in range(10):
        num_df = clean_df[clean_df.label == i]
        num_samples = len(num_df)
        num_clean = len(num_df[num_df.isClean == True])
        print("class %d : %d (%d)" % (i, num_samples, num_clean))
    
    return clean_df
    

def train(config):
    
    # Hyper-parameter settings for discriminator
    batch_size = 128
    ngpu = 1 # The number of gpu to use
    workers = 2
    lr = 0.0002
    num_epochs = 100
    beta1 = 0.5
    data_dir = ''
    device = torch.device("cuda:1" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    clean_df = get_clean_data(config)
    index_list = clean_df['index'].astype(int).tolist()
        
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=batch_size,
        shuffle=False,
        validation_split=0.0,
        num_batches=config['data_loader']['args']['num_batches'],
        training=True,
        num_workers=config['data_loader']['args']['num_workers'],
        pin_memory=config['data_loader']['args']['pin_memory'],
        sampler=torch.utils.data.SubsetRandomSampler(index_list),
        config=config
    )
    
    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)
    
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)
    
    criterion = nn.BCELoss()
    real_label = 1.
    fake_label = 0.
    fake_percent = 0.5
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    
    
    netD.train()
    print("### Start Training Discriminator ###")
    for epoch in range(num_epochs):
        losses = []
        outputs = []
        for i, (data, image_label, _, _)  in enumerate(data_loader):
            netD.zero_grad()
            
            # make noisy label (fake data) in a batch
            b_size = image_label.size(0)
            mask_label = torch.rand((b_size,), dtype=torch.float, device=device)
            mask_label[mask_label < fake_percent] = fake_label
            mask_label[mask_label >= fake_percent] = real_label
            noisy_label = torch.randint(10, (b_size,), dtype=torch.long)
            image_label[mask_label == fake_label] = noisy_label[mask_label == fake_label]
            
            # Make label tensor for cDCGAN
            label = torch.zeros(b_size, 10, 32, 32) # for CIFAR10
            for i, num in enumerate(image_label.long()):
                label[i, num, :, :] = 1
                
            data = data.to(device)
            label = label.to(device)
            output = netD(data, label).view(-1)
            errD = criterion(output, mask_label)
            errD.backward()
            
            outputs.append(output.mean().item())
            losses.append(errD.item())
            
            optimizerD.step()
            
        if epoch % 10 == 0:
            print('[%d/%d]\tLoss: %.4f' % (epoch, num_epochs, sum(losses)/len(losses)))

        if epoch == num_epochs - 1:
            disc_save_dir = 'saved/models/discriminator/'
            if not os.path.isdir(disc_save_dir):
                os.mkdir(disc_save_dir)
            model_path = disc_save_dir + time.strftime('%H%M%S') + '_best_disc.pth'
            torch.save(netD.state_dict(), model_path)
    print("### Training discriminator finished ###")
    
    return model_path
    
def test(config, model_path=None):
    
     # Hyper-parameter settings for discriminator
    batch_size = 128
    ngpu = 1 # The number of gpu to use
    workers = 2
    lr = 0.0002
    num_epochs = 200
    beta1 = 0.5
    data_dir = ''
    device = torch.device("cuda:1" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
            
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=batch_size,
        shuffle=False,
        validation_split=0.0,
        num_batches=config['data_loader']['args']['num_batches'],
        training=True,
        num_workers=config['data_loader']['args']['num_workers'],
        pin_memory=config['data_loader']['args']['pin_memory'],
        config=config
    )
    
    if model_path is None:
        model_path = 'saved/models/discriminator/154124_best_disc.pth'
    
    netD = Discriminator(ngpu).to(device)
    netD.load_state_dict(torch.load(model_path))
    netD.eval()
    
    print("### discriminator evaluation start ###")
    
    total_data = 0
    total_correct = 0
    
    with tqdm(data_loader) as progress:
        for batch_idx, (data, image_label, index, label_gt) in enumerate(progress):
            b_size = image_label.size(0)
            label = torch.zeros(b_size, 10, 32, 32) # for CIFAR10
            for i, num in enumerate(image_label.long()):
                label[i, num, :, :] = 1
                
            data = data.to(device)
            label = label.to(device)
            target = image_label == label_gt
            
            output = netD(data, label).view(-1)
            output = output >= 0.5
            
            total_data += b_size
            total_correct += torch.sum(output.cpu() == target.cpu())
            
            
    
    print("### accuracy for train data : %f ###" % (float(total_correct)/float(total_data)*100))

    
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
        CustomArgs(['--percent', '--percent'], type=float, target=('trainer', 'percent')),
        CustomArgs(['--asym', '--asym'], type=bool, target=('trainer', 'asym')),
        CustomArgs(['--name', '--exp_name'], type=str, target=('name',)),
        CustomArgs(['--seed', '--seed'], type=int, target=('seed',))
    ]
    
    config = ConfigParser.get_instance(args, options)
    if config['train_loss']['type'] == 'ELRLoss':
        options.append(CustomArgs(['--lamb', '--lamb'], type=float, target=('train_loss', 'args', 'lambda')))
        options.append(CustomArgs(['--beta', '--beta'], type=float, target=('train_loss', 'args', 'beta')))
    elif config['train_loss']['type'] == 'SCELoss':
        options.append(CustomArgs(['--alpha', '--alpha'], type=float, target=('train_loss', 'args', 'alpha')))
        options.append(CustomArgs(['--beta', '--beta'], type=float, target=('train_loss', 'args', 'beta')))
    elif config['train_loss']['type'] == 'GCELoss':
        options.append(CustomArgs(['--q', '--q'], type=float, target=('train_loss', 'args', 'q')))
        options.append(CustomArgs(['--k', '--k'], type=float, target=('train_loss', 'args', 'k')))
        options.append(CustomArgs(['--truncated', '--truncated'], type=bool, target=('train_loss', 'args', 'truncated')))

    config = ConfigParser.get_instance(args, options)

    # Set seed
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    np.random.seed(config['seed'])
    
    # Train
    model_path = train(config)
    test(config, None)
    
    
    
