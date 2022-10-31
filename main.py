from GRU_D.gru_d import GRU_D as gru_d
from SAND.model import SAnD as sand
from BRITS.brits import brits as brits
import torch
import argparse
from data_loader import get_dataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="FYP")
    parser.add_argument("--pred_window", type=float, default=0.5)
    parser.add_argument("--path", type=str, default="patient_data.csv")
    parser.add_argument('--device', default='cpu')
    parser.add_argument("--model", type=str, default="gru_d")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--type", type=str, default="test")
    parser.add_argument("--load_model", type=str, default="gru_d_0.5.pth")


    args = parser.parse_args()
    print(args)

    if args.model == "gru_d":
        model = gru_d(input_size=18, rnn_hid_size=32, device=args.device).to(args.device)
    elif args.model == "brits":
        model = brits(input_size=18, rnn_hid_size=32).to(args.device)
    elif args.model == "sand":
        model = sand(input_features=18).to(args.device)


    data_loader = get_dataloader(path=args.path, time_window=args.pred_window)


    ##------------------Only for training (ignore)--------------------
    if args.type == "train":
        epoch = args.epoch
        bce_loss = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
        for epoch_no in range(epoch):
            avg_loss = 0
            model.train()
            for batch_no, (x, y, mask, time_stamp, record_num) in enumerate(data_loader, start=1):
                x = x.to(args.device)
                y = y.to(args.device)
                mask = mask.to(args.device)
                time_stamp = time_stamp.to(args.device)
                record_num = record_num.to(args.device)
                output = model(x, mask, record_num, time_stamp)
                if args.model == "gru_d" or args.model == "sand":
                    pred_prob = output
                else:
                    pred_prob = output["predictions"]
                loss = bce_loss(pred_prob.squeeze(-1), y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                avg_loss += loss.item()

            print("epoch: ", epoch_no,
                  "Avg Loss：{:.4f}".format(avg_loss / batch_no)
                  )
        output_path = "{0}_{1}.pth".format(args.model, args.pred_window)
        torch.save(model.state_dict(), output_path)

    ##------------------For testing (load a pre-trained model)--------------------
    elif args.type == "test":
        preds = []
        for batch_no, (x, y, mask, time_stamp, record_num) in enumerate(data_loader, start=1):
            x = x.to(args.device)
            y = y.to(args.device)
            mask = mask.to(args.device)
            time_stamp = time_stamp.to(args.device)
            record_num = record_num.to(args.device)
            model.load_state_dict(torch.load("./load_model/{}".format(args.load_model)))
            model = model.to(args.device)
            output = model(x, mask, record_num, time_stamp)
            if args.model == "gru_d" or args.model == "sand":
                pred_prob = output
            else:
                pred_prob = output["predictions"]
            preds.append(pred_prob.squeeze(-1))
        preds = torch.cat(preds, dim=0)
