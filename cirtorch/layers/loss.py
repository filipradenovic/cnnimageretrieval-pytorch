import torch
import torch.nn as nn

import cirtorch.layers.functional as LF

# --------------------------------------
# Loss/Error layers
# --------------------------------------

class ContrastiveLoss(nn.Module):
    r"""Creates a criterion that measures the triplet loss given an input
    tensors x1, x2, x3 and a margin with a value greater than 0.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n`: anchor, positive examples and negative
    example respectively. The shape of all input variables should be
    :math:`(N, D)`.

    The distance swap is described in detail in the paper `Learning shallow
    convolutional feature descriptors with triplet losses`_ by
    V. Balntas, E. Riba et al.

    .. math::
        L(a, p, n) = \frac{1}{N} \left( \sum_{i=1}^N \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\} \right)

    where :math:`d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p`.

    Args:
        anchor: anchor input tensor
        positive: positive input tensor
        negative: negative input tensor
        p: the norm degree. Default: 2

    Shape:
        - Input: :math:`(N, D)` where `D = vector dimension`
        - Output: :math:`(N, 1)`

    >>> contrastive_loss = ContrastiveLoss(margin=0.7)
    >>> input = autograd.Variable(torch.randn(128, 35))
    >>> label = autograd.Variable(torch.Tensor([-1, 1, 0, 0, 0, 0, 0] * 5))
    >>> output = contrastive_loss(input, label)
    >>> output.backward()

    .. _Learning shallow convolutional feature descriptors with triplet losses:
        http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf
    """

    def __init__(self, margin=0.7, eps=1e-6):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, x, label):
        return LF.contrastive_loss(x, label, margin=self.margin, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + str(self.margin) + ')'
