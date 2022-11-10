from GRU_D.gru_d import GRU_D as gru_d
from SAND.model import SAnD as sand
from BRITS.brits import brits as brits
import torch
import argparse
from data_loader import get_dataloader
import json


def test(input):
    
    input = json.loads(input)  
    model_name = input["model_name"]
    device = input["device"]
    path = input["path"]
    load_model = input["load_model"]
    pred_window = input["pred_window"]
    
    
    
    if model_name == "gru_d":
        model = gru_d(input_size=18, rnn_hid_size=32, device=device).to(device)
    elif model_name == "brits":
        model = brits(input_size=18, rnn_hid_size=32).to(device)
    elif model_name == "sand":
        model = sand(input_features=18).to(device)

    data_loader = get_dataloader(path=path, time_window=pred_window)
    preds = []
    ids = []
    for batch_no, (x, y, mask, time_stamp, record_num, id) in enumerate(data_loader, start=1):
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        id = id.to(device)
        time_stamp = time_stamp.to(device)
        record_num = record_num.to(device)
        model.load_state_dict(torch.load("./load_model/{}".format(load_model)))
        model = model.to(device)
        model.eval()
        ids.append(id)
        output = model(x, mask, record_num, time_stamp)
        output = output.detach()
        if model_name == "gru_d" or model_name == "sand": # the output value means the risk of developing liver disease
            pred_prob = output 
        else:
            pred_prob = output["predictions"]
        
        preds.append(pred_prob.squeeze(-1))
    
    ids = torch.cat(ids, dim=0).cpu().numpy()
    preds = torch.cat(preds, dim=0).cpu().numpy() # size: B, "B" is patient number. 
    
    output = { 'pred' : preds.tolist(), 'id' : ids.tolist()}
    output = json.dumps(output)
    # print(output)
        
    return output
    

#-----Test for above function----

input = {
    "pred_window":0.5,
    "device":"cpu",
    "model_name":"gru_d",
    "load_model":"gru_d_0.5.pth",
    "path":"patient_data.csv"
    
}

# input = {
#     "pred_window":1.0,
#     "device":"cpu",
#     "model_name":"gru_d",
#     "load_model":"gru_d_1.0.pth",
#     "path":"patient_data.csv"
    
# }


# input = {
#     "pred_window":0.5,
#     "device":"cpu",
#     "model_name":"brist",
#     "load_model":"brits_0.5.pth",
#     "path":"patient_data.csv"
    
# }

# input = {
#     "pred_window":1.0,
#     "device":"cpu",
#     "model_name":"brist",
#     "load_model":"brits_1.0.pth",
#     "path":"patient_data.csv"
    
# }



input = json.dumps(input)
test(input)
    
    
    
     
