# coding: utf-8
# *****************************************************
# Aitrox Information Technology
# http://www.proxima-ai.com
# Copyright 2021 Aitrox. All rights reserved.
# CreateDate: 2021-08-25
# ******************************************************

import torch
from torch.nn import Sequential, Linear, Dropout, ReLU
from torchvision.models import densenet121


class DensNet121(torch.nn.Module):
    def __init__(self, n_classes):
        super(DensNet121, self).__init__()
        densenet = densenet121(pretrained=True)
        densenet.classifier = Sequential(Linear(1024, 256),
                                         ReLU(),
                                         Dropout(p=0.2),
                                         Linear(256, n_classes))
        self.model = densenet

    def forward(self, x):
        x = self.model(x)
        x = torch.nn.functional.softmax(x, dim=1)

        return x
