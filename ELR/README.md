# ELR
This is an non-official PyTorch implementation of ELR method proposed in [Early-Learning Regularization Prevents Memorization of Noisy Labels]().
This also include PyTorch implementation of SCE and GCE method.


## Usage
Train the network on the Symmmetric Noise CIFAR-10 dataset (noise rate = 0.8):

```
python train.py -c ./hyperparams/cosine/config_cifar10_elr.json --percent 0.8
python train.py -c ./hyperparams/multistep/config_cifar10_elr.json --percent 0.8
```

Train the network on the Asymmmetric Noise CIFAR-100 dataset (noise rate = 0.4):

```
python train.py -c ./hyperparams/cosine/config_cifar100.json --percent 0.4 --asym 1
```

The config files can be modified to adjust hyperparameters and optimization settings. 

### arguments
```
python train.py --no_wandb
```
not uses wandb, just run on your local(server)

```
python train.py --dataset cifar10 --loss_fn sce --lr_scheduler cosine
```
or
```
python train.py --config hyperparams/~~/~~~.json
```
if dataset, loss_fn, lr_scheduler are all given, don't have to give config file as an argument.
if config file is given, dataset, loss_fn, lr_scheduler arguments are useless.

## References
- S. Liu, J. Niles-Weed, N. Razavian and C. Fernandez-Granda "Early-Learning Regularization Prevents Memorization of Noisy Labels", 2020
