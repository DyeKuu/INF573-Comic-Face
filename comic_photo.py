import sys
import cv2 as cv
from image.image import TwoImages

if __name__ == "__main__":
    if len(sys.argv) != 4 and len(sys.argv) != 3:
        print(
            f'Usage: {sys.argv[0]} comic_face.png person_photo.png [-m]\nOption: -m : merge the comic face')
        sys.exit(1)

    ifMerge = len(sys.argv) == 4
    a = TwoImages(person_filename=sys.argv[2],
                  comic_filename=sys.argv[1])
    # from mtcnn import MTCNN
    # d = MTCNN()
    # a = TwoImages(person_filename=sys.argv[2],
    #               comic_filename=sys.argv[1], detector=d)
    im = a.fusion(merge=ifMerge)
    cv.imshow("Fusion", im)
    cv.waitKey(0)
    cv.destroyAllWindows()
