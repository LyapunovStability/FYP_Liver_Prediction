from GRU_D.gru_d import GRU_D as gru_d
from SAND.model import SAnD as sand
from BRITS.brits import brits as brits
import torch
import argparse
from data_loader import get_loader

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="FYP")
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--path", type=str, default="dm_2.npz")
    parser.add_argument('--device', default='cpu')
    parser.add_argument("--model", type=str, default="brits")
    parser.add_argument("--input_size", type=int, default=17)

    args = parser.parse_args()
    if args.model == "gru_d":
        model = gru_d(input_size=args.input_size, rnn_hid_size=32, device=args.device).to(args.device)
    elif args.model == "brits":
        model = brits(input_size=args.input_size, rnn_hid_size=32).to(args.device)
    elif args.model == "sand":
        model = sand(input_features=args.input_size).to(args.device)


    data_loader = get_loader()

    epoch = 1
    for epoch_no in range(epoch):
        avg_loss = 0
        model.train()
        for batch_no, (x, y, mask, time_stamp, record_num) in enumerate(data_loader, start=1):
            output = model(x, mask, record_num, time_stamp)
