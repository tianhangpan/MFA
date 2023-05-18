import torch
import torch.nn as nn
import thop
import torchsummary

from model.model import ConvNeXtIndoorR, ConvNeXtFuse


class MFA(nn.Module):
    def __init__(self):
        super().__init__()
        self.amb_model = ConvNeXtIndoorR(in_channels=6, mode=4)
        self.counter = ConvNeXtIndoorR(in_channels=3, mode=4)
        self.fuse_model = ConvNeXtFuse(mode=4)
        self.d_max = 1.5
        self.d_min = 1.0

    def amb_linea_trans(self, amb_map):
        amb_1d = amb_map.view(amb_map.size(0), 1, -1)
        amb_max = torch.max(amb_1d, dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        amb_min = torch.min(amb_1d, dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        amb_map = (amb_map - amb_min) * (self.d_max - self.d_min) / (amb_max - amb_min) + self.d_min
        return amb_map

    def forward(self, x):
        print(x.shape)
        dif = torch.randn((2, 6, 224, 224))
        _, counter_features = self.counter(x)
        amb_res, amb_features = self.amb_model(dif)
        counter_res = self.fuse_model(counter_features, amb_features)
        amb_res = self.amb_linea_trans(amb_res)
        final_res = counter_res * amb_res
        return final_res


model = MFA()
# inputs = torch.randn((1, 3, 224, 224))
print(torchsummary.summary(model, input_size=(3, 224, 224), batch_size=-1))
# print(thop.profile(model, (inputs,)))
# time_s = time.time()
# _ = model(inputs)
# print(time.time() - time_s)
