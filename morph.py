from image.morpher import morph
from image.dlib_detector import DLIB_DETECTOR

import cv2
detector = DLIB_DETECTOR()
src_im = cv2.imread("comic_pics/ki.png")
src_pts = detector.face_points(src_im)
dst_im = cv2.imread("human_pics/img.PNG")
dst_pts = detector.face_points(dst_im)
morph(src_im, src_pts, dst_im, dst_pts, background="transparent")
