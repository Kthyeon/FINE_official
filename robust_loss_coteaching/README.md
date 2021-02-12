# ELR and Co-teaching With CLK or SAME
This is an non-official PyTorch implementation of ELR method proposed in [Early-Learning Regularization Prevents Memorization of Noisy Labels]().
This also include PyTorch implementation of SCE and GCE method.

## Usage
### Robust loss functions
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

### Co-teaching families
Train the network on the Symmetric Noise CIFAR-10 dataset (noise rate = 0.8). Fix lr_scheduler argument as coteach
You can choose four arguments for loss_fn (coteach, coteach+, coteachdistill, coteach+distill)
```
python train_coteaching.py --arch=rn18 --loss_fn=coteach --lr_scheduler=coteach --num_gradual=60 --asym=False --percent=0.8 --no_wandb
python train.py --device 0 --config hyperparams/coteach/config_cifar10_coteach_rn18.json --asym=False --percent=0.8 --no_wandb
```
not uses wandb, just run on your local(server)

Train the networ with the CLK or SAME with coteachdistill or coteach+distill (noise rate = 0.8)
```
python train_coteaching.py --distillation --reinit --distill_mode kmeans --arch=rn18 --asym=False --dataset=cifar100 --loss_fn=coteach+distill --lr_scheduler=coteach --num_gradual=140 --percent=0.8 
```
or
```
python train_coteaching.py --distillation --reinit --distill_mode eigen --arch=rn18 --asym=False --dataset=cifar100 --loss_fn=coteachdistill --lr_scheduler=coteach --num_gradual=60 --percent=0.8 
```

## arguments
if dataset, loss_fn, lr_scheduler are all given, don't have to give config file as an argument.
if config file is given, dataset, loss_fn, lr_scheduler arguments are useless.

### Robust loss functions

```
usage : python train.py [-c] [-d] [--distillation] [--distill_mode] [--dataset] [--percent] [--asym] [--loss_fn] [--lr_scheduler] [--percent] [--no_wandb]

    arguments : 
        -c, --config : config file path
        -d, --device : device number
        
    options :
        --distillation : using distillation or not
        --distill_mode : SAME if eigen, CLK if kmeans
        --dataset : using dataset
        --percent : noise rate for synthetic noisy dataset
        --asym : symmetric noise if False else asymmetric noise
        --loss_fn : loss function for training model (CE, GCE, SCE, ELR, ...)
        --lr_scheduler : multistep scheduler or cosine annealing scheduler (multistep, cosine)
        --no_wandb : whether or not using wandb (if you do not use wandb, state --no_wandb)
```

### Co-teaching families

```
usage : python train.py [-c] [-d] [--distillation] [--distill_mode] [--dataset] [--percent] [--asym] [--loss_fn] [--lr_scheduler] [--percent] [--arch] [--num_gradual] 
        [--no_wandb]

    arguments : 
        -c, --config : config file path
        -d, --device : device number
        
    options :
        --distillation : using distillation or not
        --distill_mode : SAME if eigen, CLK if kmeans
        --dataset : using dataset (cifar10, cifar100)
        --percent : noise rate for synthetic noisy dataset
        --asym : symmetric noise if False else asymmetric noise
        --loss_fn : loss function for training model (CE, GCE, SCE, ELR, ...)
        --lr_scheduler : multistep scheduler or cosine annealing scheduler
        --arch : architecture for student nework
        --num_gradual : $E_{k}$ for co-teaching+ (warm-up epochs for filtering the noisy instances)
        --no_wandb : whether or not using wandb (if you do not use wandb, state --no_wandb)
```