#!/bin/bash

python main.py -d 1 --asym false --percent 0.6 --lr_scheduler multistep --arch rn34 --loss_fn cce --dataset cifar10 --traintools coteaching --no_wandb --distill_mode fine-gmm --dynamic --dataseed 123 --every 10 --warmup 40