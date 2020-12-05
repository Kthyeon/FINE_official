#!/bin/bash

noise_list="0.2 0.4 0.6 0.8"

# for noise in $noise_list
# do
#     python train.py -c ./hyperparams/multistep/config_cifar100_gce.json \
#                     --percent $noise \
#                     --run_name "cifar100_baseline_gce_sym_$noise"
# done

for noise in $noise_list
do
    python train.py -c ./hyperparams/multistep/config_cifar100_sce.json \
                    --percent $noise \
                    --run_name "cifar100_baseline_sce_sym_$noise"
done

for noise in $noise_list
do
    python train.py -c ./hyperparams/multistep/config_cifar100_elr.json \
                    --percent $noise \
                    --run_name "cifar100_baseline_elr_sym_$noise"
done

for noise in $noise_list
do
    python train.py -c ./hyperparams/multistep/config_cifar100_tgce.json \
                    --percent $noise \
                    --run_name "cifar100_baseline_tgce_sym_$noise"
done
    