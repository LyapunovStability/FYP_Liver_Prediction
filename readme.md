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



## Command for running the model

"load_model" folder: 3 pre-trained models with time window "1 year" or "0.5 year"

when running a pre-trained model---ex, GRU_D 

```key
     python main.py --pred_window 0.5  --type test --device cpu --model gru_d --load_model gru_d_0.5.pth
``` 

