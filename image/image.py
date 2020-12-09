# System package
from image.dlib_detector import DLIB_DETECTOR
import numpy as np
from mtcnn import MTCNN
import cv2 as cv
import os
from typing import Tuple
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
# Third part package


class Image():
    def __init__(self, input=None, path=None, convert=False, detector=None):
        '''
        Class to realise basic operations for images
        Input:
            * input: an image matrix
            * path: path of the image (input and path can not both be None)
            * convert: if the picture is comic, we have to set convert = True the color into BRG
                       in order to use MTCNN
        '''
        if path is not None:
            self.im = cv.imread(path)
        elif input is None:
            raise("Arg input Or path should not be both None")
        else:
            self.im = input

        self.isConverted = False
        if convert:
            self.convert()
        self.res = None  # detect result by detector
        self.detector = MTCNN() if detector is None else detector

    def detect_face(self) -> np.array:
        """
        This function will return a 2-dim np.arrary, which is a list of face feature points.
        The return value will differ according to different detector.
        """
        if isinstance(self.detector, MTCNN):
            self.res = np.array(
                list(self.detector.detect_faces(self.im)[0]["keypoints"].values()))
        elif isinstance(self.detector, DLIB_DETECTOR):
            self.res = self.detector.face_points(self.im)
        self.convert2real_image()
        print(self.res)

        return self.res

    def image_shape(self) -> Tuple[int, int, int]:
        return self.im.shape

    def convert(self):
        if self.isConverted:
            self.isConverted = False
            self.im = cv.cvtColor(self.im, cv.COLOR_BGR2RGB)
        else:
            self.isConverted = True
            self.im = cv.cvtColor(self.im, cv.COLOR_RGB2BGR)

    def convert2real_image(self):
        if self.isConverted:
            self.convert()

    def rotate_image(self, apply=False, points=False, rotate_angle=None, rotate_center=None, rotated=False):
        if rotated or isinstance(self.detector, DLIB_DETECTOR):
            if points:
                return self.res
            return self.im

        if rotate_angle is None or rotate_angle is None:
            right_eye = self.res[3]
            left_eye = self.res[2]
            nose = self.res[0]
            x = (right_eye + left_eye)/2 - nose
            Lx = np.sqrt(x.dot(x))
            cos_angle = -x[1]/Lx
            self.rotate_angle = np.arccos(cos_angle)*360/2/np.pi
            if x[0] < 0:
                self.rotate_angle = self.rotate_angle*-1
            self.rotate_center = tuple(nose)
        else:
            self.rotate_center = rotate_center
            self.rotate_angle = rotate_angle

        self.rotation_matrix = cv.getRotationMatrix2D(
            self.rotate_center, self.rotate_angle, 1.0)  # Maybe we don't need this affection

        if points:
            points = np.array(list(self.res.values()))
            ones = np.ones(shape=(len(points), 1))
            points_ones = np.concatenate((points, ones), axis=1)
            rotated_points = self.rotation_matrix.dot(points_ones.T).T

            return rotated_points
        else:
            rotated_image = cv.warpAffine(self.im, self.rotation_matrix, dsize=(
                self.im.shape[1], self.im.shape[0]))
            if apply:
                self.im = rotated_image

            return rotated_image

    def preprocess(self, path, face_path=None):
        self.detect_face()
        self.rotate_image(apply=True)
        self.convert2real_image()
        if face_path is not None:
            face = Image(path=face_path, detector=self.detector)
            face.rotate_image(rotate_angle=self.rotate_angle,
                              rotate_center=self.rotate_center, apply=True)
            cv.imwrite(face_path, face.im)
        cv.imwrite(path, self.im)


class TwoImages():
    def __init__(self,
                 person_input=None,
                 person_filename=None,
                 comic_input=None,
                 comic_filename=None,
                 detector=None):
        """
        Core Class for this package: Class to compare and fusion 2 images.
        Input:
            * person_input: a person image matrix
            * person_filename: path of the human image (input and path can not both be None)
            * comic_input: a comic image matrix
            * comic_filename: path of the comic image (input and path can not both be None)
            * detector: a face detector class
        """
        self.person_image = Image(
            input=person_input, path=person_filename, detector=detector)
        self.comic_image = Image(
            input=comic_input, path=comic_filename, convert=True, detector=detector)
        self.height, self.width, _channels = self.person_image.image_shape()
        self.comic_width = int(
            self.height / self.comic_image.image_shape()[0] * self.comic_image.image_shape()[1])
        self.comic_image.im = cv.resize(self.comic_image.im, (self.comic_width, self.height),
                                        interpolation=cv.INTER_AREA)
        self.detector = detector
        # self.comic_image.im = cv.resize(self.comic_image.im, (self.height, self.width),
        #                                interpolation=cv.INTER_AREA) # Resize to show the comparison

        self.fusion_res = None  # fusion result of those two images

    def replace_with_face(self, face_input, face_filename):
        '''
        Sometimes MTCNN can not recognise a comic image when there is only the head, so we want to find its keypoints
        by the full image and give it to the image only with face.
        Has to be used after detect_res
        Args:
            face_input: a comic face image matrix
            face_filename: path of the face image (input and path can not both be None)
            comic_res: result for a converted and resized comic image
        '''
        if face_filename is not None:
            face_im = cv.imread(face_filename)
        elif input is None:
            raise ("Arg input Or path should not be both None")
        else:
            face_im = face_input
        face_im = cv.resize(face_im, (self.width, self.height),
                            interpolation=cv.INTER_AREA)

        self.comic_image.im = face_im

    def detect_res(self):
        self.person_image.detect_face()
        self.comic_image.detect_face()
        return self.person_image.res, self.comic_image.res

    def compare(self):
        '''
        Function to show the comparison image for the human face and the comic face.
        '''
        res1, res2 = self.detect_res()
        self.comic_image.convert2real_image()

        full_image = cv.hconcat([self.person_image.im, self.comic_image.im])

        for k in range(len(res1)):
            cv.line(full_image, tuple(res1[k]), (res2[k][0] +
                                                 self.width, res2[k][1]), (0, 255, 0), thickness=2)

        return full_image

    def fusion(self, face_input=None, face_filename=None):
        '''
        Compare two images and calculate H matrix directly
        '''
        real_pts, comic_pts = self.detect_res()
        self.comic_image.convert2real_image()

        from facemorpher import locator
        from facemorpher import warper
        from facemorpher import blender
        size = (self.height, self.width)
        percent = 0.5
        points = locator.weighted_average_points(
            comic_pts, real_pts, percent)
        src_face = warper.warp_image(
            self.comic_image.im, comic_pts, points, size)
        end_face = warper.warp_image(
            self.person_image.im, real_pts, points, size)
        average_face = blender.weighted_average(src_face, end_face, percent)
        comic_pts = points
        mask = blender.mask_from_points(size, points)
        blended_img = blender.poisson_blend(
            src_face, end_face, mask)
        cv.imshow("blended", blended_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

        self.morph_image = average_face
        self.comic_image.im = blended_img

        self.M, _mask = cv.findHomography(comic_pts, real_pts, cv.RANSAC, 5.0)
        if face_input is not None or face_filename is not None:
            self.replace_with_face(face_input=face_input,
                                   face_filename=face_filename)
        dst = cv.warpPerspective(
            self.comic_image.im, self.M, (self.width, self.height))  # wraped image
        for i in range(self.height):
            if (dst[i, ] == 0).all():
                dst[i, ] = self.person_image.im[i, ]
                continue
            for j in range(self.width):
                if (dst[i, j] == 0).all():
                    dst[i, j] = self.person_image.im[i, j]

        return dst

    def fusion_rotated(self, face_input=None, face_filename=None):
        '''
        Function to optimise the fusion result.
        Precisely, we rotate our 2 images into the front, and then calculate H by two front images, and at last rotate
        to the original position.
        '''
        if isinstance(self.detector, DLIB_DETECTOR):
            return self.fusion()
        self.detect_res()
        # TODO: do not know why the result is different is we do rotation first and save image
        self.comic_image.rotate_image(apply=True)
        comic_pts = self.comic_image.rotate_image(apply=False, points=True)
        real_pts = self.person_image.rotate_image(apply=False, points=True)
        # comic_pts = self.comic_image.rotate_image(apply=False, points=True, rotated=True)

        self.M, _mask = cv.findHomography(comic_pts, real_pts, cv.RANSAC, 5.0)
        if face_input is not None or face_filename is not None:
            self.replace_with_face(face_input=face_input,
                                   face_filename=face_filename)
        dst = cv.warpPerspective(
            self.comic_image.im, self.M, (self.width, self.height))

        reverse_matrix = cv.getRotationMatrix2D(
            self.person_image.rotate_center, -self.person_image.rotate_angle, 1.0)

        dst = cv.warpAffine(dst, reverse_matrix,
                            dsize=(self.width, self.height))

        for i in range(self.height):
            if (dst[i, ] == 0).all():
                dst[i, ] = self.person_image.im[i, ]
                continue
            for j in range(self.width):
                if (dst[i, j] == 0).all():
                    dst[i, j] = self.person_image.im[i, j]

        return dst

    def run_fusion(self, rotate=True, merge=False, face_input=None, face_filename=None):
        if merge:
            self.transfer_color(apply=True)
        if not rotate:
            return self.fusion(face_input=face_input, face_filename=face_filename)
        else:
            return self.fusion_rotated(face_input=face_input, face_filename=face_filename)

    @staticmethod
    def image_stats(image):
        # compute the mean and standard deviation of each channel
        (l, a, b) = cv.split(image)
        (lMean, lStd) = (l.mean(), l.std())
        (aMean, aStd) = (a.mean(), a.std())
        (bMean, bStd) = (b.mean(), b.std())

        # return the color statistics
        return (lMean, lStd, aMean, aStd, bMean, bStd)

    def transfer_color(self, apply=False):
        # Define target and source
        source = self.person_image.im
        target = self.comic_image.im
        source = cv.cvtColor(source, cv.COLOR_BGR2LAB).astype("float32")
        target = cv.cvtColor(target, cv.COLOR_BGR2LAB).astype("float32")
        (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc,
         bStdSrc) = TwoImages.image_stats(source)
        (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar,
         bStdTar) = TwoImages.image_stats(target)
        # subtract the means from the target image
        (l, a, b) = cv.split(target)
        l -= lMeanTar
        a -= aMeanTar
        b -= bMeanTar
        # scale by the standard deviations
        l = (lStdTar / lStdSrc) * l
        a = (aStdTar / aStdSrc) * a
        b = (bStdTar / bStdSrc) * b
        # add in the source mean
        l += lMeanSrc
        a += aMeanSrc
        b += bMeanSrc
        # clip the pixel intensities to [0, 255] if they fall outside
        # this range
        l = np.clip(l, 0, 255)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)
        # merge the channels together and convert back to the RGB color
        # space, being sure to utilize the 8-bit unsigned integer data
        # type
        transfer = cv.merge([l, a, b])
        transfer = cv.cvtColor(transfer.astype("uint8"), cv.COLOR_LAB2BGR)
        # show and return the color transferred image
        cv.imshow("transfer", transfer)
        cv.waitKey(0)
        cv.destroyAllWindows()

        tmp = self.comic_image.im.copy()
        tmp[tmp != 0] = transfer[tmp != 0]

        if apply:
            self.comic_image.im = tmp

        return tmp
