# Sample-Selection Approaches and Conjunction with Noise-Robust functions
This is a PyTorch implementation for the sample-selection approaches and conjunction with noise-robust functions.

## Usage
You can check simple descriptions about arguments in `utils/args.py`.
According to the descriptions, the arguments can be replaced.

All the bash samples below run the code with 60% symmetric noise, cifar-10 dataset and ResNet-34 architecture.
You can change arguments settings according to its descriptions.

### FINE as robust approach (Sec. 4.2.3)
Dynamically apply FINE algorithm in the training process.
Bash files for this section is in `scripts/dynamic/` directory.

```
bash scripts/dynamic/FINE_ce_dynamic.sh
bash scripts/dynamic/FINE_gce_dynamic.sh
bash scripts/dynamic/FINE_sce_dynamic.sh
bash scripts/dynamic/FINE_elr_dynamic.sh

```

### Robust loss functions
Train the proxy network on the Symmmetric Noise CIFAR-10 dataset, ResNet18, ELR Loss (noise rate = 0.8):

```
python train.py -c ./hyperparams/multistep/config_cifar10_elr_rn18.json --percent=0.8 --asym=False
```

Train the proxy network on the Asymmmetric Noise CIFAR-100 dataset, ResNet34, GCE Loss (noise rate = 0.4):

```
python train.py -c ./hyperparams/cosine/config_cifar100_gce_rn34.json --percent=0.4 --asym=True
```

If you want to train the network with SAME CIFAR-10 dataset, GCELoss, ResNet34(noise rate = 0.8, Symmetric Noise),

```
train.py -c ./hyperparams/multistep/config_cifar10_gce_rn34.json -d 0 --percent 0.8 --distillation --distill_mode=eigen 
--load_name=multistep_sym_80_gce.pth --reinit
```
To load the checkpoint for `--load_name`, you should manually make the `checkpoint` folder and put the `xx.pth` file into the `checkpoint` folder.
(`xx.pth` will be saved in the `saved` directory and its log is in the `logger`.)
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
To load the checkpoint for `--load_name`, you should manually make the `checkpoint` folder and put the `xx.pth` file into the `checkpoint` folder.
(`xx.pth` will be saved in the `saved` directory and its log is in the `logger`.)
## arguments
if dataset, loss_fn, lr_scheduler are all given, don't have to give config file as an argument.
if config file is given, dataset, loss_fn, lr_scheduler arguments are useless.

### Robust loss functions

```
usage : python train.py [-c] [-d] [--distillation] [--distill_mode] [--dataset] [--percent] [--asym] [--loss_fn] [--lr_scheduler] 
                        [--percent] [--no_wandb] [--reinit] [--load_name] [--mode]

    arguments : 
        -c, --config : config file path
        -d, --device : device number
        
    options :
        --distillation : using distillation or not
        --distill_mode : SAME if eigen, CLK if kmeans
        --dataset : using dataset
        --percent : noise rate for synthetic noisy dataset
        --asym : symmetric noise if False else asymmetric noise
        --loss_fn : loss function for training model (cce, gce, sce, elr)
        --lr_scheduler : multistep scheduler or cosine annealing scheduler (multistep, cosine)
        --no_wandb : whether or not using wandb (if you do not use wandb, state --no_wandb)
        --reinit : whether or not re-initialization network parameters
        --load_name : checkpoint directory for proxy network
        --mode : traning with same loss as proxy network if same, training with ce loss if ce
```

### Co-teaching families

```
usage : python train_coteaching.py [-c] [-d] [--distillation] [--distill_mode] [--dataset] [--percent] [--asym] [--loss_fn] [--lr_scheduler] [--percent] [--arch] 
                                    [--num_gradual] [--no_wandb] [--reinit] [--load_name]

    arguments : 
        -c, --config : config file path
        -d, --device : device number
        
    options :
        --distillation : using distillation or not
        --distill_mode : SAME if eigen, CLK if kmeans
        --dataset : using dataset (cifar10, cifar100)
        --percent : noise rate for synthetic noisy dataset
        --asym : symmetric noise if False else asymmetric noise
        --loss_fn : loss function for training model (coteach, coteach+, coteachdistill, coteach+distill)
        --lr_scheduler : scheduler for learning rate (coteach)
        --arch : architecture for student nework
        --num_gradual : $E_{k}$ for co-teaching+ (warm-up epochs for filtering the noisy instances)
        --no_wandb : whether or not using wandb (if you do not use wandb, state --no_wandb)
        --reinit : whether or not re-initialization network parameters
        --load_name : checkpoint directory for proxy network
```