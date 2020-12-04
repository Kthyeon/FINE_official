#!/bin/bash

noise_list="0.1 0.2 0.3 0.4"

for noise in $noise_list
do
    python train.py -c ./hyperparams/multistep/config_cifar100_gce.json \
                    --percent $noise \
                    --project "cifar100_baseline_asym_$noise" 
done

for noise in $noise_list
do
    python train.py -c ./hyperparams/multistep/config_cifar100_sce.json --percent $noise
                    --percent $noise \
                    --project "cifar100_baseline_asym_$noise" 
done

for noise in $noise_list
do
    python train.py -c ./hyperparams/multistep/config_cifar100_elr.json --percent $noise
                    --percent $noise \
                    --project "cifar100_baseline_asym_$noise" 
done

for noise in $noise_list
do
    python train.py -c ./hyperparams/multistep/config_cifar100_tgce.json --percent $noise
                    --percent $noise \
                    --project "cifar100_baseline_asym_$noise" 
done
    