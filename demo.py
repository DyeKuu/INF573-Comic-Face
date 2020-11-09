import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

from mtcnn import MTCNN
import cv2

def compare_faces(filename1, filename2):
    im1 = cv2.cvtColor(cv2.imread(filename1), cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(cv2.imread(filename2), cv2.COLOR_BGR2RGB)
    dim = (1280, 1280)
    im1 = cv2.resize(im1, dim, interpolation=cv2.INTER_AREA)
    im2 = cv2.resize(im1, dim, interpolation=cv2.INTER_AREA)
    full_image = cv2.hconcat([im1,im2])

    detector = MTCNN()
    res1 = detector.detect_faces(im1)[0]['keypoints']
    res2 = detector.detect_faces(im2)[0]['keypoints']

    for k in res1:
        cv2.line(full_image, res1[k], res2[k] + dim[0], (0, 255, 0), thickness=2)

    cv2.imshow("Face comparaison", full_image)

    return full_image

res = compare_faces("img.PNG", "ki.png")
cv2.imshow("comparaison", res)


