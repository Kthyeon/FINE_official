#!/bin/bash
python Train_cifar.py --gpuid 0 --noise_mode asym --r 0.4 --distill dynamic --lambda_u 0
python Train_cifar.py --gpuid 0 --noise_mode sym --r 0.2 --distill dynamic --lambda_u 0
python Train_cifar.py --gpuid 0 --noise_mode sym --r 0.5 --distill dynamic --lambda_u 25
python Train_cifar.py --gpuid 0 --noise_mode sym --r 0.8 --distill dynamic --lambda_u 25
python Train_cifar.py --gpuid 0 --noise_mode sym --r 0.9 --distill dynamic --lambda_u 50