import torch
import torch.nn as nn

import cirtorch.layers.functional as LF

# --------------------------------------
# Normalization layers
# --------------------------------------

class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N,self).__init__()
        self.eps = eps

    def forward(self, x):
        return LF.l2n(x, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'


class PowerLaw(nn.Module):

    def __init__(self, eps=1e-6):
        super(PowerLaw, self).__init__()
        self.eps = eps

    def forward(self, x):
        return LF.powerlaw(x, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'