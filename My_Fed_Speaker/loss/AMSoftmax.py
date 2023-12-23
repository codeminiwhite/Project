
import torch
import torch.nn as nn
from  Utils.tools import *

class AMSoftmax(nn.Module):
    def __init__(self, embedding_size, n_class, m, s):
        super(AMSoftmax, self).__init__()

        self.m = m
        self.s = s
        self.W = torch.nn.Parameter(torch.randn(embedding_size, n_class), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, label=None):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)

        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)

        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()

        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.to(costh.device)
        #
        costh_m = costh - delt_costh
        output = self.s * costh_m

        loss = self.ce(output, label.to(torch.long))
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, prec1
