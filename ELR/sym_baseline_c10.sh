#!/bin/bash

noise_list="0.2 0.4 0.6 0.8"

for noise in $noise_list
do
    python train.py -c ./hyperparams/multistep/config_cifar10_gce.json \
                    --percent $noise \
                    --project "cifar10_baseline_sym_$noise" 
done

for noise in $noise_list
do
    python train.py -c ./hyperparams/multistep/config_cifar10_sce.json --percent $noise
                    --percent $noise \
                    --project "cifar10_baseline_sym_$noise" 
done

for noise in $noise_list
do
    python train.py -c ./hyperparams/multistep/config_cifar10_elr.json --percent $noise
                    --percent $noise \
                    --project "cifar10_baseline_sym_$noise" 
done

for noise in $noise_list
do
    python train.py -c ./hyperparams/multistep/config_cifar10_tgce.json --percent $noise
                    --percent $noise \
                    --project "cifar10_baseline_sym_$noise" 
done
    