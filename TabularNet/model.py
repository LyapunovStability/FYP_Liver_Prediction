import torch
import torch.nn as nn
from torch.autograd import Function


def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H

class GLU(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, in_size)
        self.linear2 = nn.Linear(in_size, in_size)

    def forward(self, X):
        return self.linear1(X) * self.linear2(X).sigmoid()

class Feature_Extractor(nn.Module):

    def __init__(self, num_dim, cat_dim):
        super(Feature_Extractor, self).__init__()

        self.hid_dim = 16
        self.emb_dim = 16
        self.set_num(with_DM=with_DM, with_num_25=with_num_25)
        self.num_dim = num_dim
        self.cat_dim = cat_dim

        self.num_f_emb = nn.Sequential(nn.Linear(self.num_dim + self.static_val, self.emb_dim), GLU(self.emb_dim), nn.Dropout(0.1))
        self.cat_f_emb = nn.Sequential(nn.Linear(self.cat_num, self.emb_dim), GLU(self.emb_dim), nn.Dropout(0.1))
        self.mask_emb = nn.Sequential(nn.Linear(self.num_dim, self.emb_dim), GLU(self.emb_dim), nn.Dropout(0.1))
        self.all_emb = nn.Sequential(nn.Linear(self.emb_dim * 3, self.hid_dim * 2), GLU(self.hid_dim * 2), nn.Linear(self.hid_dim * 2, self.hid_dim), GLU(self.hid_dim), nn.Dropout(0.1))


    def forward(self, num_data, cat_data, mask):

        f1 = self.num_f_emb(num_data)
        f2 = self.cat_f_emb(cat_data)
        f_mask = self.mask_emb(mask)
        f = torch.cat([f1, f2, f_mask], dim=-1)
        f = self.all_emb(f)


        return f



class Classifier(nn.Module):

    def __init__(self, feature_dim, with_DM=True, with_num_25=True):
        super(Classifier, self).__init__()

        if with_num_25:
            self.hid_dim = 16
        else:
            self.hid_dim = 16


        self.classifier = nn.Linear(self.hid_dim, 1)


    def forward(self, feature):

        class_output = self.classifier(feature)

        return class_output


class Predictor(nn.Module):

    def __init__(self, feature_extractor=None, classifier=None):
        super(Predictor, self).__init__()


        self.feature_extractor = feature_extractor
        self.classifier = classifier


    def forward(self, num_data, cat_data, mask):

        feature = self.feature_extractor(num_data, cat_data, mask)
        class_output = self.classifier(feature)

        return class_output






class Imputer(nn.Module):

    def __init__(self, feature_dim, with_DM=True, with_num_25=True):
        super(imputer, self).__init__()

        self.hid_dim = 16
        self.num_f_emb = nn.Sequential(nn.Linear(self.param_num, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, self.param_num))


    def forward(self, x, mask):
        x_num = x[:, :self.param_num]
        x_num = x_num * mask
        x_recon = self.num_f_emb(x_num)
        x[:, :self.param_num] = x[:, :self.param_num] * mask + (1 - mask) * x_recon

        return x


