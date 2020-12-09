# Third part package
from image.dlib_detector import DLIB_DETECTOR
import cv2 as cv
# Inside import
from image.image import Image, TwoImages
from mtcnn import MTCNN


class Video():
    '''
    Class to replace human face by a comic image in a video.
    '''

    def __init__(self,
                 video_path=None,
                 comic_path=None, detector=None):
        self.video = cv.VideoCapture(video_path)
        self.comic_image = Image(path=comic_path)
        self.cur_human_image = None
        self.fps = self.video.get(cv.CAP_PROP_FPS)
        self.size = (int(self.video.get(cv.CAP_PROP_FRAME_WIDTH)),
                     int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT)))
        self.detector = DLIB_DETECTOR() if detector is None else detector

    def process_video(self):
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        videoWriter = cv.VideoWriter(
            'results/output.avi', fourcc, self.fps, self.size)
        success, frame = self.video.read()
        videoWriter.write(frame)
        while success:
            self.cur_human_image = frame
            try:
                a = TwoImages(person_input=frame,
                              comic_input=self.comic_image.im, detector=self.detector)
                im = a.fusion_rotated()
                videoWriter.write(im)
                print(im.shape)
                print(frame.shape)
                cv.imshow("new video", im)
                cv.waitKey(1000 / int(self.fps))
                cv.waitKey()
                success, frame = self.video.read()
            except:
                success, frame = self.video.read()
                continue

        self.video.release()

        return videoWriter
