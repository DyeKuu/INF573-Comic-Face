import cv2 as cv


def test_TwoImages():
    from image import TwoImages
    a = TwoImages(person_filename="img.PNG", comic_filename="ki.png")
    im = a.compare()
    cv.imshow("Face comparaison", im)
    cv.waitKey(0)
    cv.destroyAllWindows()


def test_rotateImage():
    from image import Image
    a = Image(path="ki.png", convert=True)
    a.detect_face()
    cv.imshow("Rotation", a.rotate())
    cv.waitKey(0)
    cv.destroyAllWindows()


def test_fusion():
    from image import Image, TwoImages
    a = TwoImages(person_filename="img.PNG", comic_filename="ki.png")
    im = a.fusion()
    cv.imshow("Fusion", im)
    cv.waitKey(0)
    cv.destroyAllWindows()


def test_fusion_rotated():
    from image import Image, TwoImages
    a = TwoImages(person_filename="img.PNG", comic_filename="ki.png")
    im = a.fusion_rotated()
    cv.imshow("Fusion_rotated", im)
    cv.waitKey(0)
    cv.destroyAllWindows()


test_fusion_rotated()
