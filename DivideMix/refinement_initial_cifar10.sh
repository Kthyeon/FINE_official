#!/bin/bash
python Train_cifar_robustloss.py --gpuid 1 --noise_mode asym --r 0.4 --distill initial --teacher_model pretrained/multistep_asym_40_elr.pth --lambda_u 0 --refinement
python Train_cifar_robustloss.py --gpuid 1 --noise_mode sym --r 0.2 --distill initial --teacher_model pretrained/multistep_asym_40_elr.pth --refinement --lambda_u 0 --refinement
python Train_cifar_robustloss.py --gpuid 1 --noise_mode sym --r 0.8 --distill initial --teacher_model pretrained/multistep_asym_40_elr.pth --refinement --lambda_u 25 --refinement
