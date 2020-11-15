import cv2 as cv


def test_TwoImages():
    from image.image import TwoImages
    a = TwoImages(person_filename="human_pics/img.PNG", comic_filename="comic_pics/ki.png")
    im = a.compare()
    cv.imshow("Face comparaison", im)
    cv.waitKey(0)
    cv.destroyAllWindows()


def test_rotateImage():
    from image.image import Image
    a = Image(path="comic_pics/ki.png", convert=True)
    a.detect_face()
    cv.imshow("Rotation", a.rotate_image())
    cv.waitKey(0)
    cv.destroyAllWindows()


def test_fusion():
    from image.image import TwoImages
    a = TwoImages(person_filename="human_pics/img.PNG", comic_filename="comic_pics/ki.png")
    im = a.fusion()
    cv.imshow("Fusion", im)
    cv.waitKey(0)
    cv.destroyAllWindows()


def test_fusion_rotated():
    from image.image import TwoImages
    a = TwoImages(person_filename="human_pics/img.PNG", comic_filename="comic_pics/ki.png")
    im = a.fusion_rotated()
    cv.imshow("Fusion_rotated", im)
    cv.waitKey(0)
    cv.destroyAllWindows()


def test_TwoImages_with_face():
    from image.image import TwoImages
    a = TwoImages(person_filename="human_pics/img.PNG", comic_filename="comic_pics/sensei2.jpg")
    im = a.compare()
    cv.imshow("Face comparaison", im)
    cv.waitKey(0)
    cv.destroyAllWindows()


def test_with_face():
    from image.image import TwoImages
    a = TwoImages(person_filename="human_pics/img.PNG", comic_filename="comic_pics/sensei2.jpg")
    im = a.fusion(face_filename="comic_pics/sensei.png")
    cv.imshow("Fusion_rotated", im)
    cv.waitKey(0)
    cv.destroyAllWindows()


def test_with_face_rotated():
    from image.image import TwoImages
    a = TwoImages(person_filename="human_pics/img.PNG", comic_filename="comic_pics/sensei2.jpg")
    im = a.fusion_rotated(face_filename="comic_pics/sensei.png")
    cv.imshow("Fusion_rotated", im)
    cv.waitKey(0)
    cv.destroyAllWindows()


test_with_face_rotated()
