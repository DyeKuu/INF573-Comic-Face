"""
::

  Morph from source to destination face or
  Morph through all images in a folder

  Usage:
    morpher.py (--src=<src_path> --dest=<dest_path> | --images=<folder>)
              [--width=<width>] [--height=<height>]
              [--num=<num_frames>] [--fps=<frames_per_second>]
              [--out_frames=<folder>] [--out_video=<filename>]
              [--plot] [--background=(black|transparent|average)]

  Options:
    -h, --help              Show this screen.
    --src=<src_imgpath>     Filepath to source image (.jpg, .jpeg, .png)
    --dest=<dest_imgpath>   Filepath to destination image (.jpg, .jpeg, .png)
    --images=<folder>       Folderpath to images
    --width=<width>         Custom width of the images/video [default: 500]
    --height=<height>       Custom height of the images/video [default: 600]
    --num=<num_frames>      Number of morph frames [default: 20]
    --fps=<fps>             Number frames per second for the video [default: 10]
    --out_frames=<folder>   Folder path to save all image frames
    --out_video=<filename>  Filename to save a video
    --plot                  Flag to plot images to result.png [default: False]
    --background=<bg>       Background of images to be one of (black|transparent|average) [default: black]
    --version               Show version.
"""

import os
import numpy as np
import cv2

from facemorpher import locator
from facemorpher import aligner
from facemorpher import warper
from facemorpher import blender
from facemorpher import plotter


def load_image_points(path, size):
    img = cv2.imread(path)
    points = locator.face_points(img)

    if len(points) == 0:
        print('No face in %s' % path)
        return None, None
    else:
        return aligner.resize_align(img, points, size)


def load_valid_image_points(imgpaths, size):
    for path in imgpaths:
        img, points = load_image_points(path, size)
        if img is not None:
            print(path)
            yield (img, points)


def morph(src_img, src_points, dest_img, dest_points, percent=0.5,
          width=500, height=600, num_frames=20, fps=10,
          out_frames=None, out_video=None, plot=False, background='black'):
    """
    Create a morph sequence from source to destination image

    :param src_img: ndarray source image
    :param src_points: source image array of x,y face points
    :param dest_img: ndarray destination image
    :param dest_points: destination image array of x,y face points
    :param video: facemorpher.videoer.Video object
    """
    size = (height, width)
    plt = plotter.Plotter(plot, num_images=num_frames, out_folder=out_frames)

    plt.plot_one(src_img)
    #video.write(src_img, 1)
    red = [0, 0, 255]

    # Produce morph frames!

    points = locator.weighted_average_points(
        src_points, dest_points, percent)
    src_face = warper.warp_image(src_img, src_points, points, size)
    end_face = warper.warp_image(dest_img, dest_points, points, size)
    average_face = blender.weighted_average(src_face, end_face, percent)

    # for point in points:
    #     print(point)
    #     average_face[point[1], point[0]] = red
    if background in ('transparent', 'average'):
        mask = blender.mask_from_points(average_face.shape[:2], points)
        average_face = np.dstack((average_face, mask))

    plt.plot_one(average_face)
    plt.save(average_face, filename=percent)
    cv2.imwrite(str(percent)+".png", average_face)
    cv2.imshow(str(percent), average_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # video.write(average_face)

    plt.plot_one(dest_img)
    #video.write(dest_img, stall_frames)
    plt.show()
