import torch
import torch.nn as nn



class CharnonnierLoss(nn.Module):
    def __init__(self,eps=1e-6):
        super(CharnonnierLoss,self).__init__()
        self.eps = eps

    def forward(self,x,y):
        diff = x-y
        loss = torch.sum(torch.sqrt(diff*diff+self.eps**2))/20
        return loss