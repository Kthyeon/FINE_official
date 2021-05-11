#!/bin/bash

python main.py -d 0 --distillation --load_name rn34/multistep_sym_80_elr.pth --asym False --percent 0.8 --lr_scheduler multistep --arch rn34 --loss_fn elr --dataset cifar10 --mode keeploss --traintools robustloss --distill_mode kmeans