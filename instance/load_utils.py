import numpy
import torch

from evolveface.backbone.model_irse import IR_50


def loadModel(modelPath, device):
    """
    Load Backbone from given path
    :param modelPath: Backbone model path
    :return: Loaded backbone
    """

    backbone = IR_50([112, 112]).to(device)
    backbone.load_state_dict(torch.load(modelPath))
    backbone.to(device)
    return backbone


def loadFaceData(facePath, device):
    """
    Load facedata by given absolute directory
    :param facePath: Absolute directory contains .npy files
    :return:
    """
    npFeatures = numpy.load(facePath + "features.npy")
    npLabels = numpy.load(facePath + "labels.npy")

    featureTensors = []

    # Break down in to tensors
    for featuresPerLabel in npFeatures:
        featureTensorsPerLabel = []
        for image in featuresPerLabel:
            featureTensorsPerLabel.append(torch.from_numpy(image).to(device))
        featureTensors.append(featureTensorsPerLabel)

    labels = npLabels.tolist()

    return featureTensors, labels
