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
- scikit-learn




## Usage
The 3 pre-trained models with two kinds of prediction time window ("1 year" or "0.5 year") are saved in the "load_model" folder 

python function ("test()" in the main.py) for testing above pre-trained models(ex, GRU-D):


```bash
# GRU-D, Prediction_Window = 0.5 year
from main import test
import json
input = {
    "pred_window":0.5,
    "device":"cpu",
    "model_name":"gru_d",
    "load_model":"gru_d_0.5.pth",
    "path":"patient_data.csv"
    
}
input = json.dumps(input)
test(input)
```

```bash
# GRU-D, Prediction_Window = 1.0 year
from main import test
import json
input = {
    "pred_window":1.0,
    "device":"cpu",
    "model_name":"gru_d",
    "load_model":"gru_d_1.0.pth",
    "path":"patient_data.csv"
    
}
input = json.dumps(input)
test(input)


``` 

## Q & A
**Q1:** *What does the output of model mean ?*

**A1:** The output of model is a value of [0, 1], and it caaould represent the risk of developing liver disease for this patient. For example, if the output is 0.19, we could assume that the model think this patient will have 19% probability to develop liver disease. 

