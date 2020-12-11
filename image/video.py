# Third part package
from image.dlib_detector import DLIB_DETECTOR
import cv2 as cv
# Inside import
from image.image import Image, TwoImages
import pyvirtualcam
import numpy as np


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

    def process_video(self, show_process=False):
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
                im = a.run()
                videoWriter.write(im)
                if show_process:
                    cv.imshow("new video", im)
                    cv.waitKey(1000 / int(self.fps))
                    cv.waitKey()
                success, frame = self.video.read()
            except:
                if show_process:
                    print("failed")
                # videoWriter.write(frame)
                success, frame = self.video.read()
                continue

        self.video.release()

        return videoWriter


class VirtualCamera:
    def __init__(self,
                 comic_path=None,
                 detector=None):
        self.detector = DLIB_DETECTOR() if detector is None else detector
        self.virtual_camera = pyvirtualcam.Camera(width=640, height=480, fps=2)
        initial_image = np.zeros(
            (self.virtual_camera.height, self.virtual_camera.width, 3), np.uint8)
        self.comparer = TwoImages(person_input=initial_image,
                                  comic_input=cv.imread(comic_path), detector=self.detector)
        self.camera = cv.VideoCapture(0)

    def run(self,
            rotate=False,
            merge=True,
            merge_color=False,
            face_input=None,
            face_filename=None):
        debug = False

        while True:
            ret, frame = self.camera.read()
            self.comparer.reset_person(person_input=frame)
            im = np.zeros((self.virtual_camera.height,
                           self.virtual_camera.width, 4), np.uint8)  # RGBA
            im[:, :, 3] = 255
            try:
                frame = self.comparer.run(rotate=rotate,
                                          merge=merge,
                                          merge_color=merge_color,
                                          face_input=face_input,
                                          face_filename=face_filename)
                if debug:
                    print(self.comparer.person_image.isConverted)
                    print(self.comparer.comic_image.isConverted)
            except:
                print('failed')

            im[:, :, :3] = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            self.virtual_camera.send(im)
            self.virtual_camera.sleep_until_next_frame()
            self.virtual_camera.sleep_until_next_frame()
