from time import *

import cv2 as cv
from PIL import Image

from evolveface.align.detector import detect_faces
from evolveface.align.visualization_utils_opencv import show_result

cap = cv.VideoCapture(cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv.CAP_PROP_FPS, 30)
flag = cap.isOpened()
font = cv.FONT_HERSHEY_SIMPLEX
while flag:
    timeStart = time()
    flag, mat = cap.read()
    if flag:

        img = Image.fromarray(cv.cvtColor(mat, cv.COLOR_BGR2RGB))
        boundingBoxes, landmarks = detect_faces(img)
        boxedMat = show_result(mat, boundingBoxes, landmarks)
        timeEnd = time()
        elapse = round(1.0 / (timeEnd - timeStart), 2)
        cv.putText(boxedMat, str(elapse), (5, 30), font, 1.2, (255, 0, 0), 2)
        cv.imshow("Result", boxedMat)

        if cv.waitKey(5) == 27:
            break
