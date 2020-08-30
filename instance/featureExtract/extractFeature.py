import argparse
import os

import cv2 as cv
import numpy
import torch
from tqdm import tqdm

from evolveface.util.extract_feature_v3 import extract_feature
from instance.load_utils import loadModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face feature extract")
    parser.add_argument("-src_root", "--src_root", help="specify your source dir", default="../database/Alpha/",
                        type=str)
    parser.add_argument("-dst_root", "--dst_root", help="specify your destination dir", type=str,
                        default="../database/AlphaFeature/")
    parser.add_argument("-model_path", "--model_path", help="Specify model dir", type=str,
                        default="../../model/backbone_ir50_ms1m_epoch120.pth")
    args = parser.parse_args()

    srcRoot = args.src_root
    dstRoot = args.dst_root
    modelPath = args.model_path
    backbone = loadModel(modelPath, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    backbone.eval()

    if not os.path.isdir(dstRoot):
        os.mkdir(dstRoot)

    labels = []
    features = []
    featuresPerLabel = []

    for subFolder in tqdm(os.listdir(srcRoot)):
        for image in os.listdir(os.path.join(srcRoot, subFolder)):
            mat = cv.imread(os.path.join(srcRoot, subFolder, image))
            try:
                feature = extract_feature(mat, backbone, original=0)
                print("Processing:{}".format(subFolder + '/' + image))
            except Exception:
                print("Warning:{}".format(Exception.message))
            featuresPerLabel.append(feature.numpy())
            if featuresPerLabel:
                features.append(featuresPerLabel)
                label = numpy.char.asarray(subFolder)
                labels.append(label)
        label = []
        featuresPerLabel = []

    npLabels = numpy.array(labels)
    npFeatures = numpy.array(features)

    numpy.save(dstRoot + "features.npy", npFeatures)
    numpy.save(dstRoot + "labels.npy", npLabels)
