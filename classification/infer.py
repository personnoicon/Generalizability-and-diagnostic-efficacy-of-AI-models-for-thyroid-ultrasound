# coding: utf-8
# *****************************************************
# Aitrox Information Technology
# http://www.proxima-ai.com
# Copyright 2021 Aitrox. All rights reserved.
# CreateDate: 2021-08-26
# ******************************************************

import os
import torch
import numpy as np
from PIL import Image
from src.models import DensNet121
from src.uniform_scale import UniformScale


try:
    print('--loading model--')
    torch.cuda.set_device(1)
    device = torch.device('cuda')
    pth_file = ''
    input_classes = 2
    state = torch.load(pth_file, map_location='cpu')
    model = DensNet121(input_classes).cuda()
    model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
    model.eval()
    print('--model load sucessed--')
except Exception as e:
    print('--initilize model error:{}--'.format(e))


def predict(img_full_path):
    input_size = 512
    img = Image.open(img_full_path)
    img_np = UniformScale(input_size, img).scale()
    img_to_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    tensor_batch = torch.unsqueeze(img_to_tensor, 0).cuda()
    output = torch.squeeze(model(tensor_batch)).tolist()
    
    return output


if __name__ == '__main__':
    res = predict(img_full_path='')
                
               
        