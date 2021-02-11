# DivideMix With Winning Tickets
We add some commands for either CLK or SAME by refering to the original code from https://github.com/LiJunnan1992/DivideMix.

## Training 
### Baseline
Run the followings:

    bash scripts/baseline_c10.sh
    bash scripts/baseline_c100.sh


### With CLK or SAME
Run the followings:

    bash scripts/refinement_dynamic_cifar10
    
### Training options

    --refinement

if not uses this option, it means that it sets 'clean probability' as one for all clean subset.

    --distill dynamic

use this to reproduce our paper. If not, DivideMix baseline training.

    --distill_mode eigen or kmeans

set which noisy detect algorithms to use, which proposed in our paper.


<b>License</b>\
This project is licensed under the terms of the MIT license.
