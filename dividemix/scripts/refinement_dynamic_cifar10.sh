#!/bin/bash
# For CLK
python Train_cifar.py --gpuid 1 --noise_mode sym --r 0.2 --distill dynamic --refinement --lambda_u 0 --distill_mode kmeans
python Train_cifar.py --gpuid 1 --noise_mode sym --r 0.8 --distill dynamic --refinement --lambda_u 25 --distill_mode kmeans

# For SAME
python Train_cifar.py --gpuid 1 --noise_mode sym --r 0.2 --distill dynamic --refinement --lambda_u 0 --distill_mode eigen
python Train_cifar.py --gpuid 1 --noise_mode sym --r 0.8 --distill dynamic --refinement --lambda_u 25 --distill_mode eigen
