# Liver Disease Prediction

## Model for Clinical Time Series Data


* BRITS
    * Paper --- BRITS: bidirectional recurrent imputation for time series
    * Link --- https://proceedings.neurips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html


* GRU-D
    * Paper --- Recurrent neural networks for multivariate time series with missing values
    * Link ---  https://arxiv.org/abs/1606.01865

* SanD
    * Paper --- Attend and diagnose: Clinical time series analysis using attention models
    * Link ---  https://arxiv.org/abs/1711.03905

## Requirements

- Python 3.9
- numpy == 1.19.0
- pandas == 0.25.1
- torch == 1.9.0



## Usage
The 3 pre-trained models with two kinds of prediction time window ("1 year" or "0.5 year") are saved in the "load_model" folder 

Command for testing above pre-trained models(ex, GRU-D):


```bash

# GRU-D, Prediction_Window = 0.5 year
python main.py --pred_window 0.5  --type test --device cpu --model gru_d --load_model gru_d_0.5.pth

# GRU-D, Prediction_Window = 1.0 year
python main.py --pred_window 1.0  --type test --device cpu --model gru_d --load_model gru_d_1.0.pth

``` 

