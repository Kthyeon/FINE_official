# Winning Ticket in Noisy Image Classification
This repository is the official implementation of "Winning Ticket in Noisy Classificaition" paper.

## Reference Codes
We refer to some official implementation codes

 - https://github.com/bhanML/Co-teaching
 - https://github.com/LiJunnan1992/DivideMix
 - https://github.com/shengliu66/ELR

 




## Requirements
- This codebase is written for `python3`.
- To install necessary python packages, run `pip install -r requirements.txt`.


## Training
### DivideMix
- Most codes are similar with the original implementation code in https://github.com/LiJunnan1992/DivideMix. So, if you want to run the baseline model, just run the `baseline.sh` file in the `dividemix` folder.
- If you want to train the model with either `CLK` or `SAME`, move to the folder `dividemix` and run the bash files by following the `README.md` file in the `dividemix` folder.

### Robust loss and Co-teaching
 - Most codes are similar with the original implementation code in https://github.com/bhanML/Co-teaching and https://github.com/shengliu66/ELR. If you want to run the baseline models, run the `robust_loss_baseline.sh` file or `co-teaching_baseline.sh` file.
 - If you want to train the model with either `CLK` or `SAME`, move to the folder `robust_loss_coteaching` and run the bash files by following the `README.md` file in the `robust_loss_coteaching` folder.

## Results
You can reproduce all results in the paper with our code. All results have been described in our paper including Appendix. The results of our experiments are so numerous that it is difficult to post everything here. However, if you experiment several times by modifying the hyperparameter value in the .sh file, you will be able to reproduce all of our analysis.
