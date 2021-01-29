#!/bin/bash
python Train_cifar_robustloss.py --gpuid 1 --noise_mode asym --r 0.4 --distill initial --refinement --lambda_u 0
python Train_cifar_robustloss.py --gpuid 1 --noise_mode sym --r 0.2 --distill initial --refinement --lambda_u 0
python Train_cifar_robustloss.py --gpuid 1 --noise_mode sym --r 0.8 --distill initial --refinement --lambda_u 25
