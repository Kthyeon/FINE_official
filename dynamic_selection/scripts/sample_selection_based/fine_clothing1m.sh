#!/bin/bash

python main.py -d 1 --lr_scheduler clothing1m --arch rn50 --loss_fn cce --dataset clothing1m --no_wandb --traintools trainingclothing1m --distill_mode fine-gmm --warmup 1 --every 1