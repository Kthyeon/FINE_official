python Train_cifar.py --gpuid 1 --noise_mode sym --r 0.9 --lambda_u 50 --distill_mode fine-kmeans  --distill dynamic --p_threshold 0.6

python Train_cifar.py --gpuid 0 --noise_mode sym --r 0.8 --lambda_u 25 --distill_mode fine-kmeans --distill dynamic --p_threshold 0.5