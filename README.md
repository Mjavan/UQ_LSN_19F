# UQ_LSN_19F
This repository provides implimentation in PyTorch for the paper "Quantifying model uncertainty for semantic segmentation of Fluorine-19 MRI using stochastic gradient MCMC". 

![Alt text](Full_image.png)

**Approach**: The paper integrates a Bayesian approach to the Unet for semnatic segentation of low signal-to-noise ratio (SNR) of 19F MRI. 19F MRI is a reall dataset that detecting 19F MRI signals is importnat for studying varoius diseases and treatments. However the low signal-to-noise ratio (SNR) of 19F MRI necessitates computational methods to reliably detect 19F signal regions and segment these from the background. In this paper we posrpoed a Bayesian Deep Learning model that not only increases sensitivity of 19F MRI significantly but also it provides uncertainty maps to detct the prediction failures and investigating the regions with high uncertainties using experts. 

**Accepted in: Computer Vision and Image Understanding**

https://doi.org/10.1016/j.cviu.2024.103967







### Usage:

To train the model, run the code in `train.py`. You can simply run the following:

`python UQ_LSN_19F/train.py`

It will run code with default parameters, and save chekpoints in `ckpts` folder.

**Parameters**:


To test the model, use `test.py`. You can simply run the following code:

`python UQ_LSN_19F/test.py`



