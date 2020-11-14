import cv2 as cv
from mtcnn import MTCNN
import os
import numpy as np
from typing import Type
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


class Image():
    def __init__(self, input=None, path=None, convert=False):
        if path is not None:
            self.im = cv.imread(path)
        elif input is None:
            raise("Arg input Or path should not be both None")
        else:
            self.im = input

        self.isConverted = False
        if convert:
            self.convert_image()
        self.res = None

    def detect_face(self):
        detector = MTCNN()
        self.res = detector.detect_faces(self.im)[0]["keypoints"]
        print(self.res)

        return self.res

    def image_shape(self):
        return self.im.shape

    def convert_image(self):
        if self.isConverted:
            self.isConverted = False
            self.im = cv.cvtColor(self.im, cv.COLOR_BGR2RGB)
        else:
            self.isConverted = True
            self.im = cv.cvtColor(self.im, cv.COLOR_RGB2BGR)


class TwoImages():
    def __init__(self, person_input=None, person_filename=None, comic_input=None, comic_filename=None):
        """
        Load image and resize
        """
        self.PersonImage = Image(person_input, person_filename)
        self.ComicImage = Image(comic_input, comic_filename, convert=True)
        height, width, _channels = self.PersonImage.image_shape()
        self.ComicImage.im = cv.resize(self.ComicImage.im, (width, height),
                                       interpolation=cv.INTER_AREA)

    def detect_res(self):
        return self.PersonImage.detect_face(), self.ComicImage.detect_face()

    def compare(self):
        res1, res2 = self.detect_res()
        _height, width, _channels = self.PersonImage.image_shape()
        if self.ComicImage.isConverted:
            self.ComicImage.convert_image()
        full_image = cv.hconcat([self.PersonImage.im, self.ComicImage.im])

        for k in res1:
            cv.line(full_image, res1[k], (res2[k][0] +
                                          width, res2[k][1]), (0, 255, 0), thickness=2)
        return full_image


def rotate_image(image, res):
    right_eye = np.array(res['right_eye'])
    left_eye = np.array(res['left_eye'])
    nose = np.array(res['nose'])
    x = (right_eye + left_eye)/2 - nose
    Lx = np.sqrt(x.dot(x))
    cos_angle = x[0]/Lx
    angle = np.arccos(cos_angle)*360/2/np.pi
    M = cv.getRotationMatrix2D(tuple(nose), angle, 1.0)  # 12
    rotated = cv.warpAffine(image, M, image.shape)  # 13

    return rotated
