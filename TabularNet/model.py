import torch
import torch.nn as nn




class model(nn.Module):

    def __init__(self, input_features = 18):
        super(model, self).__init__()

        self.hid_dim = 32
        self.feature_dim = input_features * 2
        self.all_emb = nn.Sequential(nn.Linear(self.feature_dim, self.hid_dim * 2), nn.ReLU(), nn.Linear(self.hid_dim * 2, self.hid_dim), nn.ReLU(), nn.Dropout(0.1))
        self.classifier = nn.Linear(self.hid_dim, 1)

    def forward(self, x, mask, record_num, time_stamp):
        # print(x.shape)
        # print(mask.shape)
        x = torch.cat([x, mask], dim=-1)
        f = self.all_emb(x)
        pred = self.classifier(f)
        pred = torch.sigmoid(pred)

        return pred




