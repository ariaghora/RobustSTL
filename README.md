# RobustSTL: A Robust Seasonal-Trend Decomposition Algorithm for Long Time Series (AAAI 2019)
This repository contains python (3.8) implementation of RobustSTL ([paper](https://arxiv.org/abs/1812.01767))  . 

Decomposing complex time series into trend, seasonality, and remainder components is an important task to facilitate time series anomaly detection and forecasting.  
RobustSTL extract trend using LAD loss with sparse regularization and non-local seasonal filtering.  
Compared to previous approaches (such as traditional STL), RobustSTL has advantages on  
1) Ability to handle seasonality fluctuation and shift, and abrupt change in trend and reminder  
2) robustness of data with anomalies  
3) applicability on time series with long seasonality period.  

## Installation
`pip install --upgrade git+https://github.com/ariaghora/RobustSTL.git`

## Usage
```python
from rstl import RobustSTL

result = RobustSTL(input,
                   season_len,
                   reg1=10,
                   reg2=0.5,
                   K=2,
                   H=5,
                   dn1=1,
                   dn2=1,
                   ds1=50,
                   ds2=1,
                   learning_rate=0.1,
                   max_iter=100,
                   max_trial=10,
                   verbose=True)
```

## Arugments of RobustSTL
- input : input series
- season_len : length of seasonal period
- reg1: first order regularization parameter for trend extraction
- reg2: second order regularization parameter for trend extraction
- K: number of past season samples in seasonaility extraction
- H: number of neighborhood in seasonality extraction
- dn1, dn2 : hyperparameter of bilateral filter in denoising step.
- ds1, ds2 : hypterparameter of bilarteral filter in seasonality extraction step.
- learning_rate: the Adam optimizer learning rate
- max_iter: number of iterations for the Adam optimizer
- max_trials: number of outer STL iterations
- verbose: whether showing or hiding progress bar

## Shape of input sample
Basically, RobustSTL is for univariate time series sample.  
However, this codes are available on multi-variate time series sample.
(It apply the algorithm to each series, using multiprocessing)
Each series *have to* have same time length.

* Univariate Time Series: `[Time] or [Time,1]`
* Multivariate Time Series: `[N, Time] or [N, Time, 1]`

## Codes
* `example.py` : run example code
* `run_example.ipynb` : run example code in jupyter notebook

## Etc
The original paper has wrong notation in seasonality extraction.  
The difference is [log](https://github.com/LeeDoYup/RobustSTL/commit/99a801525eca59469b0a314dd17fdd798c477c6d)
