import cv2 as cv


def test_TwoImages():
    from image.image import TwoImages
    a = TwoImages(person_filename="human_pics/img.PNG",
                  comic_filename="comic_pics/ki.png")
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
    a = TwoImages(person_filename="human_pics/img.PNG",
                  comic_filename="comic_pics/ki.png")
    im = a.fusion()
    cv.imshow("Fusion", im)
    cv.waitKey(0)
    cv.destroyAllWindows()


def test_fusion_rotated():
    from image.image import TwoImages
    a = TwoImages(person_filename="human_pics/img.PNG",
                  comic_filename="comic_pics/ki.png")
    im = a.fusion_rotated()
    cv.imshow("Fusion_rotated", im)
    cv.waitKey(0)
    cv.destroyAllWindows()


def test_TwoImages_with_face():
    from image.image import TwoImages
    a = TwoImages(person_filename="human_pics/img.PNG",
                  comic_filename="face_pics/sensei.png")
    im = a.compare()
    cv.imshow("Face comparaison", im)
    cv.waitKey(0)
    cv.destroyAllWindows()


def test_with_face():
    from image.image import TwoImages
    a = TwoImages(person_filename="human_pics/img.PNG",
                  comic_filename="face_pics/sensei.png")
    im = a.fusion(face_filename="face_pics/face_sensei.png")
    cv.imshow("Fusion_rotated", im)
    cv.waitKey(0)
    cv.destroyAllWindows()


def test_with_face_rotated():
    from image.image import TwoImages
    a = TwoImages(person_filename="human_pics/img.PNG",
                  comic_filename="face_pics/sensei.png")
    im = a.fusion_rotated(face_filename="face_pics/face_sensei.png")
    cv.imshow("Fusion_rotated", im)
    cv.imwrite("results/sensei.png", im)
    cv.waitKey(0)
    cv.destroyAllWindows()


def preprocess():
    import os
    from image.image import Image
    g = os.walk(r"comic_pics")
    for path, dir_list, file_list in g:
        for file_name in file_list:
            path_tmp = os.path.join(path, file_name)
            print(path_tmp)
            a = Image(path=path_tmp, convert=True)
            a.preprocess(path_tmp)
    g = os.walk(r"face_pics")
    for path, dir_list, file_list in g:
        for file_name in file_list:
            path_tmp = os.path.join(path, file_name)
            print(path_tmp)
            print("face" in file_name)
            if "face" in file_name:
                continue
            a = Image(path=path_tmp, convert=True)
            face_path = path + r"/face_" + file_name
            a.preprocess(path_tmp, face_path=face_path)


def test_video():
    from image.video import Video
    a = Video(video_path="video/test_Trim.mp4", comic_path="comic_pics/ki.png")
    a.process_video()


def test_color_transfer():
    from image.image import TwoImages
    import cv2 as cv
    a = TwoImages(person_filename="human_pics/img.PNG",
                  comic_filename="comic_pics/ki2.png")
    im = a.run_fusion(rotate=True, merge=True)
    cv.imshow("rotate and merge", im)
    cv.imwrite("results/ki_merge.png", im)
    cv.waitKey(0)
    cv.imwrite("result/color_transfer.png", im)


test_video()
