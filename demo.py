import cv2 as cv
from mtcnn import MTCNN
import os
import numpy as np
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


def compare_faces(filename1, filename2):
    im1 = cv.imread(filename1)
    height, width, channels = im1.shape

    im2 = cv.cvtColor(cv.imread(filename2), cv.COLOR_BGR2RGB)
    im2 = cv.resize(im2, (width, height), interpolation=cv.INTER_AREA)
    print(im2.shape)
    detector = MTCNN()
    res1 = detector.detect_faces(im1)[0]['keypoints']
    res2 = detector.detect_faces(im2)[0]['keypoints']
    full_image = cv.hconcat([im1, im2])

    for k in res1:
        cv.line(full_image, res1[k], (res2[k][0] +
                                      width, res2[k][1]), (0, 255, 0), thickness=2)

    # cv.imshow("Face comparaison", full_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return im1, cv.cvtColor(im2, cv.COLOR_BGR2RGB), res1, res2


im1, im2, real_pts, comic_pts = compare_faces("human_pics/img.PNG", "comic_pics/ki2.png")
height, width, channels = im1.shape
print(np.array(list(real_pts.values())))
# print(real_pts, comic_pts)

M, mask = cv.findHomography(np.array(list(comic_pts.values())), np.array(
    list(real_pts.values())), cv.RANSAC, 5.0)

# cv.warpPerspective(im1, np.eye(3), (width, height))
# cv.warpPerspective(	im2, M, ())

dst = cv.warpPerspective(
    im2, M, (im1.shape[1], im1.shape[0]))  # wraped image
for i in range(height):
    if (dst[i, ] == 0).all():
        dst[i, ] = im1[i, ]
        continue
    for j in range(width):
        # if not dst[i, j].all(0):
        #     dst[i, j] = im1[i, j]
        if (dst[i, j] == 0).all():
            dst[i, j] = im1[i, j]
        # print(temp)
# now paste them together
# dst[0:im1.shape[0], 0:im1.shape[1]] = im1
cv.imshow("Face comparaison", dst)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite("results/fusion.jpg", dst)
