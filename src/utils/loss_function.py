# Loss function 
import torch
import numpy as np

class R2Loss():
    """
    target_medians: ??
    """
    def __init__(
        self, 
        target_medians: np.array, 
        eps: int    
    ):
        self.target_medians = target_medians
        self.eps = eps
    
    def __call__(
        self, 
        y_pred, 
        y_true
    ):
        y_median = torch.tensor(self.target_medians).to('cuda')
        eps_cuda = torch.tensor([self.eps]).to('cuda')
        ss_res = (y_true - y_pred)**2
        ss_total = (y_true - y_median)**2
        loss = torch.sum(ss_res, dim = 0) / torch.maximum(torch.sum(ss_total, dim=0), eps_cuda)
        return torch.mean(loss)