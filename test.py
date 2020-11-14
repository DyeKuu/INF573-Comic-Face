def test_TwoImages():
    from image import TwoImages
    import cv2 as cv
    a = TwoImages(person_filename="img.PNG", comic_filename="ki.png")
    im = a.compare()
    cv.imshow("Face comparaison", im)
    cv.waitKey(0)
    cv.destroyAllWindows()


test_TwoImages()
