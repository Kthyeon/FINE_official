# Sample-Selection Approaches and Collaboration with Noise-Robust functions
This is a PyTorch implementation for the sample-selection approaches and collaboration with noise-robust functions.

## Dataset
### CIFAR10, CIFAR100
You don't have to take care about these dataset. Download options of these datasets are included in the codes.

### Clothing1M
You have to download Clothing1M dataset and set its path before run the codes.
- To download the dataset, follow https://github.com/Cysu/noisy_label
- Directories and Files of clothing1m should be saved in `dir_to_data/clothing1m`. The directory structure should be

        dynamic_selection/dir_to_data/clothing1m/
        ├── 0/
        ├── ⋮
        ├── 9/
        ├── annotations/
        ├── category_names_chn.txt
        ├── category_names_eng.txt
        ├── clean_train_key_list.txt
        ├── clean_val_key_list.txt
        ├── clean_test_key_list.txt
        ├── clean_label_kv.txt
        ├── noisy_train_key_list.txt
        └── noisy_label_kv.txt

- Directories `0/` to `9/` include image data.

## Usage
You can check simple descriptions about arguments in `utils/args.py`.

All the bashes below run the code with `60% symmetric noise`, `cifar-10 dataset` and `ResNet-34` architecture.
You can change arguments settings according to its descriptions.

### Arguments setting
You can check the description of each arguments in `utils/args.py`.

However, when you execute a command, if you give dataset, lr_scheduler, loss_fn arguments manually (e.g. `python main.py --lr_scheduler multistep --loss_fn elr --dataset cifar10`), then its corresponding config file is used automatically from `hyperparams` directory.

If you give the config file manually as an argument (e.g. `python main.py --config [config_file]`), then other arguments are over-written on the given config file.


### Sample-Selection based Approaches (Sec. 4.2.1)

To run our FINE algorithm, the FINE detector dynamically select the clean data at every epoch, and then the neural network are trained with them

```
bash scripts/sample_selection_based/fine_cifar.sh

bash scripts/sample_selection_based/fine_clothing1m.sh
```
- You can change cifar10 or cifar100 option in `fine_cifar.sh`
- Clothing1m dataset have to be set before run `fine_clothing1m.sh`

To run `F-coteaching` experiment, substituting sample selection state of Co-teaching to our FINE algorithm, just follow

```
bash scripts/sample_selection_based/f-coteaching.sh
```

### Collaboration with Noise-Robust Loss Functions (Sec. 4.2.3)

These commands run our FINE algorithm with various robust loss function methods.
We used Cross Entropy (CE), Generalized Cross Entropy (GCE), Symmetric Cross Entropy (SCE), and Early-Learning Regularized (ELR).

```
bash scripts/robust_loss/fine_dynamic_ce.sh

bash scripts/robust_loss/fine_dynamic_gce.sh

bash scripts/robust_loss/fine_dynamic_sce.sh

bash scripts/robust_loss/fine_dynamic_elr.sh
```

<b>License</b>\
This project is licensed under the terms of the MIT license.


<!-- 


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
``` -->