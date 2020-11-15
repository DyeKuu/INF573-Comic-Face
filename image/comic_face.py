import cv2 as cv
from mtcnn import MTCNN
import os
import numpy as np
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

class ComicFace():
    def __init__(self, filename1, filename2,
                 path=None):
        if path is not None:
            filename1 = path
        im2 = cv.imread(filename2)