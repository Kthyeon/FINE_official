# DivideMix With Winning Tickets
We add some commands for either CLK or SAME by refering to the original code from https://github.com/LiJunnan1992/DivideMix.

## Training
To run scripts or python file, you have to download cifar-10 or cifar-100 datasets, firstly.

### Baseline
Run the followings:

    bash scripts/baseline_c10.sh
    bash scripts/baseline_c100.sh


### With CLK or SAME
All hyper-parameter settings are same with baselines
Run the followings:

    bash scripts/refinement_dynamic_cifar10
    
### arguments
Default arguments settings are set for cifar10 experiments. Usage and other arguments are same with DivideMix original code. Options below are added only for our paper.

```
usage : python Train_cifar.py [--distill] [--distill_mode] [--refinement]

    options :
        --distill : if "dynamic", using our noisy detector.
        --distill_mode [eigen, kmeans] : Which method to use for noisy detector. "eigen" uses SAME, "kmeans" uses CLK
        --refinement : if not, the clean probability of all clean subset is set to one. In our paper, all experiments always use this option to make same condition with original paper.
```


<b>License</b>\
This project is licensed under the terms of the MIT license.
