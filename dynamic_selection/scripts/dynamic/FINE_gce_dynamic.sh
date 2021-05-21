#!/bin/bash

python main.py -d 1 --asym false --percent 0.6 --lr_scheduler multistep --arch rn34 --loss_fn gce --dataset cifar10 --traintools robustloss --no_wandb --dynamic --distill_mode fine-gmm --dataseed 123 --every 5 --warmup 50