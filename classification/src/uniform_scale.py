# coding: utf-8
# *****************************************************
# Aitrox Information Technology
# http://www.proxima-ai.com
# Copyright 2021 Aitrox. All rights reserved.
# CreateDate: 2021-08-25
# ******************************************************

import numpy as np
from PIL import Image


class UniformScale:
    def __init__(self, size, img):
        """
        :param size: img size (size, size)
        :param img: pillow format img
        """
        self.size = size
        self.img = img

    def scale(self):
        img = self.img
        input_size = self.size
        img = img.convert('RGB')
        ratio = min(input_size / img.height, input_size / img.width)
        scale_img = img.resize((int(img.width * ratio), int(img.height * ratio)))
        scale_img = np.asarray(scale_img)
        pad_img = np.zeros((input_size, input_size, 3))
        dw = int((input_size - int(img.width * ratio)) / 2)
        dh = int((input_size - int(img.height * ratio)) / 2)
        pad_img[dh:int(img.height * ratio) + dh, dw:int(img.width * ratio) + dw] = scale_img
        pad_img = Image.fromarray(np.uint8(pad_img))
        pad_img = np.array(pad_img, dtype=np.uint8)

        return pad_img
