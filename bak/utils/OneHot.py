import torch
import torch.nn as nn

class OneHot(nn.Module):

    def __init__(self, outputSize):
        super(OneHot, self).__init__()
        self.outputSize = outputSize
        # We'll construct one-hot encodings by using the index method to reshuffle the rows of an
        # identity matrix. To avoid recreating it every iteration we'll cache it
        self._eye = torch.eye(outputSize)

    def forward(self, input):

