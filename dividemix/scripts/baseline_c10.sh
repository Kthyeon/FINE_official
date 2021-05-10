#!/bin/bash
python Train_cifar.py --gpuid 0 --noise_mode asym --r 0.4 --lambda_u 0
python Train_cifar.py --gpuid 0 --noise_mode sym --r 0.2 --lambda_u 0
python Train_cifar.py --gpuid 0 --noise_mode sym --r 0.5 --lambda_u 25
python Train_cifar.py --gpuid 0 --noise_mode sym --r 0.8 --lambda_u 25
