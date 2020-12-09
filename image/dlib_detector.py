import cv2
import numpy as np
import os.path as path
import dlib
import os


class DLIB_DETECTOR():

    def __init__(self):
        DATA_DIR = os.environ.get(
            'DLIB_DATA_DIR',
            path.join(path.dirname(path.dirname(
                path.realpath(__file__))), 'data')
        )
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.dlib_predictor = dlib.shape_predictor(
            path.join(DATA_DIR, 'shape_predictor_68_face_landmarks.dat'))

    def boundary_points(self, points, width_percent=0.1, height_percent=0.1):
        """ Produce additional boundary points
        :param points: *m* x 2 array of x,y points
        :param width_percent: [-1, 1] percentage of width to taper inwards. Negative for opposite direction
        :param height_percent: [-1, 1] percentage of height to taper downwards. Negative for opposite direction
        :returns: 2 additional points at the top corners
        """
        x, y, w, h = cv2.boundingRect(np.array([points], np.int32))
        spacerw = int(w * width_percent)
        spacerh = int(h * height_percent)
        return [[x+spacerw, y+spacerh],
                [x+w-spacerw, y+spacerh]]

    def face_points(self, img, add_boundary_points=True):
        return self.face_points_dlib(img, add_boundary_points)

    def face_points_dlib(self, img, add_boundary_points=True):
        """ Locates 68 face points using dlib (http://dlib.net)
        Requires shape_predictor_68_face_landmarks.dat to be in face_morpher/data
        Download at: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        :param img: an image array
        :param add_boundary_points: bool to add additional boundary points
        :returns: Array of x,y face points. Empty array if no face found
        """
        try:
            points = []
            rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rects = self.dlib_detector(rgbimg, 1)

            if rects and len(rects) > 0:
                # We only take the first found face
                shapes = self.dlib_predictor(rgbimg, rects[0])
                points = np.array([(shapes.part(i).x, shapes.part(i).y)
                                   for i in range(68)], np.int32)

                if add_boundary_points:
                    # Add more points inwards and upwards as dlib only detects up to eyebrows
                    points = np.vstack([
                        points,
                        self.boundary_points(points, 0.1, -0.03),
                        self.boundary_points(points, 0.13, -0.05),
                        self.boundary_points(points, 0.15, -0.08),
                        self.boundary_points(points, 0.33, -0.12)])

            return points
        except Exception as e:
            print(e)
            return []
