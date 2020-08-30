from time import *

import cv2 as cv
import torch
from PIL import Image

from evolveface.align.detector import detect_faces
from evolveface.align.visualization_utils_opencv import show_result
from evolveface.util.extract_feature_v3 import get_embeddings
from evolveface.util.perf_utils import lp_wrapper
from instance.calcDistance import calcDistance
from instance.load_utils import loadModel, loadFaceData


@lp_wrapper()
def Recognize():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load Reco Model
    backbone = loadModel("C:/Users/Bill Kerman/PycharmProjects/FaceNetCustomized/model/backbone_ir50_ms1m_epoch120.pth",
                         device)
    backbone.eval()

    # Load Face Data
    features, labels = loadFaceData("C:/Users/Bill Kerman/PycharmProjects/FaceNetCustomized/database/AlphaFeature/",
                                    device)

    cap = cv.VideoCapture(cv.CAP_DSHOW)
    flag = cap.isOpened()
    font = cv.FONT_HERSHEY_SIMPLEX
    while flag:
        timeStart = time()
        flag, mat = cap.read()
        if flag:
            img = Image.fromarray(cv.cvtColor(mat, cv.COLOR_BGR2RGB))
            boundingBoxes, landmarks = detect_faces(img)
            boxedMat = show_result(mat, boundingBoxes, landmarks)

            # Recognize faces
            for boundingBox in boundingBoxes:
                cropMat = mat[int(boundingBox[1]):int(boundingBox[3]), int(boundingBox[0]):int(boundingBox[2])]
                feature = get_embeddings(cropMat, backbone, device)
                dist, label = calcDistance(srcFeature=feature, features=features, labels=labels, device=device)
                if dist < 1:
                    cv.putText(mat, label + str(round(float(1 - dist) * 100.0, 2)) + "%",
                               (int(boundingBox[0]), int(boundingBox[1])), font, 0.8, (255, 255, 255), 2)
            timeEnd = time()
            elapse = round(1.0 / (timeEnd - timeStart), 2)
            cv.putText(boxedMat, str(elapse), (5, 30), font, 1.2, (255, 0, 0), 2)
            cv.imshow("Result", boxedMat)
            if cv.waitKey(5) == 27:
                break


Recognize()
