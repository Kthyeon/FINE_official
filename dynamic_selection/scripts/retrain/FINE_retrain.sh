#!/bin/bash

python main.py -c hyperparams/multistep/config_cifar10_cce_rn34.json --no_wandb -d 1 --asym false --percent 0.6 
python main.py -c ./saved/models/cifar10/resnet34/MultiStepLR/CCELoss/sym/60/config_123.json --distillation --distill_mode=fine-gmm --load_name=./saved/models/cifar10/resnet34/MultiStepLR/CCELoss/sym/60/model_best123.pth --mode=keeploss -d 1 --reinit --no_wandb --TFT --dataseed 123