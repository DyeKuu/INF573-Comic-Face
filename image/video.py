# Third part package
import cv2 as cv
# Inside import
from image.image import Image, TwoImages

def video_processor():
    '''
    Class to replace human face by a comic image in a video.
    '''
    def __init__(self,
                 video_filename,
                 comic_filename,
                 video_path=None,
                 comic_path=None):
        if video_path is not None:
            video_filename = video_path + r"/" + video_filename
        if comic_path is not None:
            comic_filename = comic_path + r"/" + comic_filename
        self.video = cv.VideoCapture(video_filename)
        self.comic_image = Image(path=comic_filename)
        self.fps = self.video.get(cv.CAP_PROP_FPS)

    def process_video(self):
        videoWriter = cv.VideoWriter('trans.mp4', cv.VideoWriter_fourcc(*'MP4V'), fps, size)
        success, frame = self.video.read()
        videoWriter.write(frame)
        while success:
            success, frame = self.video.read()
            videoWriter.write(frame)
        self.video.release()

        return videoWriter