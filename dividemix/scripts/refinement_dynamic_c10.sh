python Train_cifar.py --gpuid 0 --noise_mode sym --r 0.8 --lambda_u 150 --dataset cifar100 --num_class 100 --distill_mode fine-gmm  --distill dynamic --data_path ./cifar-100

python Train_cifar.py --gpuid 0 --noise_mode sym --r 0.9 --lambda_u 150 --dataset cifar100 --num_class 100  --p_threshold 0.6 --distill_mode fine-gmm  --distill dynamic --data_path ./cifar-100