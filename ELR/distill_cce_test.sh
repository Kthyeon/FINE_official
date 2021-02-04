#!/bin/bash
python train.py -d 1 --distillation --load_name cifar10/ResNet34/multistep_sym_80_cce.pth --asym False --percent 0.8 --lr_scheduler multistep --arch rn34 --loss_fn cce --dataset cifar10 --mode ce
python train.py -d 1 --distillation --load_name cifar10/ResNet34/multistep_sym_80_gce.pth --asym False --percent 0.8 --lr_scheduler multistep --arch rn34 --loss_fn gce --dataset cifar10 --mode ce
python train.py -d 1 --distillation --load_name cifar10/ResNet34/multistep_sym_80_sce.pth --asym False --percent 0.8 --lr_scheduler multistep --arch rn34 --loss_fn sce --dataset cifar10 --mode ce
python train.py -d 1 --distillation --load_name cifar10/ResNet34/multistep_sym_80_elr.pth --asym False --percent 0.8 --lr_scheduler multistep --arch rn34 --loss_fn elr --dataset cifar10 --mode ce
python train.py -d 1 --distillation --load_name cifar10/ResNet34/multistep_asym_40_cce.pth --asym True --percent 0.4 --lr_scheduler multistep --arch rn34 --loss_fn cce --dataset cifar10 --mode ce
python train.py -d 1 --distillation --load_name cifar10/ResNet34/multistep_asym_40_sce.pth --asym True --percent 0.4 --lr_scheduler multistep --arch rn34 --loss_fn sce --dataset cifar10 --mode ce
python train.py -d 1 --distillation --load_name cifar10/ResNet34/multistep_asym_40_gce.pth --asym True --percent 0.4 --lr_scheduler multistep --arch rn34 --loss_fn gce --dataset cifar10 --mode ce
