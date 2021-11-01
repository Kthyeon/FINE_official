# [Official] FINE Samples for Learning with Noisy Labels
This repository is the official implementation of "FINE Samples for Learning with Noisy Labels" paper presented in NeurIPS 2021. New version of previous repository https://github.com/jaychoi12/FINE. Future code modifications and official developments will take place here. Thanks to the contributors in the previous repo.

- Paper, NeurIPS 21, FINE Samples for Learning with Noisy Labels, [Arxiv](https://arxiv.org/abs/2102.11628)

## Reference Codes
We refer to some official implementation codes

 - https://github.com/bhanML/Co-teaching
 - https://github.com/LiJunnan1992/DivideMix
 - https://github.com/shengliu66/ELR

 
## Requirements
- This codebase is written for `python3` (used `python 3.7.6` while implementing).
- To install necessary python packages, run `pip install -r requirements.txt`.


## Training

### Sample-Selection Approaches and Collaboration with Noise-Robust loss functions
 - Most code strucutres are similar with the original implementation code in https://github.com/bhanML/Co-teaching and https://github.com/shengliu66/ELR. 
 - If you want to train the model with `FINE`, move to the folder `dynamic_selection` and run the bash files by following the `README.md`.
 
### Semi-Supervised Approaches
- Most codes are similar with the original implementation code in https://github.com/LiJunnan1992/DivideMix. 
- If you want to train the model with `FINE` (`f-dividemix`), move to the folder `dividemix` and run the bash files by following the `README.md` in the `dividemix` folder.


## Results
You can reproduce all results in the paper with our code. All results have been described in our paper including Appendix. The results of our experiments are so numerous that it is difficult to post everything here. However, if you experiment several times by modifying the hyperparameter value in the .sh file, you will be able to reproduce all of our analysis.

## Contact
 - Jongwoo Ko : Jongwoo.ko@kaist.ac.kr
 - Taehyeon Kim : potter32@kaist.ac.kr

<b>License</b>\
This project is licensed under the terms of the MIT license.

## Acknowledgements
This work was supported by Institute of Information & communications Technology Planning &
Evaluation (IITP) grant funded by the Korea government (MSIT) \[No.2019-0-00075, Artificial Intelligence Graduate School Program (KAIST)] and \[No. 2021-0-00907, Development of Adaptive and Lightweight Edge-Collaborative Analysis Technology for Enabling Proactively Immediate Response and Rapid Learning].
