# Semi-Supervised Learning based Approaches
Overall code structure is made from original DivideMix repo.
We refer to https://github.com/LiJunnan1992/DivideMix.

## Dataset
### CIFAR10, CIFAR100
To download CIFAR dataset, just follow

```
bash scripts/download_cifar10.sh

bash scripts/download_cifar100.sh
```

### Clothing1M
Please follow https://github.com/jaychoi12/FINE/tree/master/dynamic_selection clothing1m settings.
You can set data path as an argument.

## Training
To run scripts or python file, you have to download cifar-10 or cifar-100 datasets, firstly.


### With FINE
All hyper-parameter settings are the same with baselines
Run the followings for severe noise rate 90\%:
```
    bash scripts/refinement_dynamic_cifar10
    bash scripts/refinement_dynamic_cifar100
```
<!-- ### arguments
Default arguments settings are set for cifar10 experiments. Usage and other arguments are same with DivideMix original code. 


 -->
<b>License</b>\
This project is licensed under the terms of the MIT license.
