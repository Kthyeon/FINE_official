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

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Discriminator should be modified to utilize conditional variable (image label)
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(64 * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size. 1 x 1 x 1
        )

    def forward(self, input, label):
        return self.main(input, label)
    
def main():
    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    # Hyper-parameter settings
    batch_size = 128
    ngpu = 1 # The number of gpu to use
    workers = 2
    lr = 0.0002
    num_epochs = 20
    beta1 = 0.5
    data_dir = ''
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    dataset = dset.ImageFolder(root=data_dir,
                              transform=None # Need to add target transformation to make fake data
                              )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
    
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
    fake_percent = 0.4
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    
    losses = []
    
    print("Start Training Discriminator")
    for epoch in range(num_epochs):
        for i, data, image_label  in enumerate(dataloader, 0):
            netD.zero_grad()
            
            data = data.to(device)
            image_label = image_label.to(device)
            
            # make noisy label (fake data) in a batch
            b_size = image_label.size(0)        
            mask_label = torch.rand((b_size,), dtype=torch.float, device=device)
            mask_label[mask_label < fake_percent] = fake_label
            mask_label[mask_label >= fake_percent] = real_label
            noisy_label = torch.randint(10, (b_size,), dtype=torch.float)
            image_label[mask_label == fake_label] = noisy_label[mask_label == fake_label]
            
            
            output = netD(data, image_label).view(-1)
            errD = criterion(output, mask_label)
            errD.backward()
            
            D_x = output.mean().item()
            
            optimizerD.step()
            
            if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss: %.4f\tD(x): %.4f'
                  % (epoch, num_epochs, i, len(dataloader), errD.item(), D_x))
            
            losses.append(errD.item())

            
    
if __name__ == '__main__':
    main()

    
    
    
