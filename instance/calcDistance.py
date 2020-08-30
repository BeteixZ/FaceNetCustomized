import numpy
import torch


def calcDistance(srcFeature, features, labels, device):
    '''
    Calculate the minimal distance between source face and face dataset
    :param srcFeature: Single tensor contains source feature
    :param features: Feature tensor list from the database
    :param labels: Labels of the feature list
    :returns Minimal distance and the predicted label of source feature
    '''
    index = 0
    minDist = numpy.inf
    for i, featureTensor in enumerate(features):
        for image in featureTensor:
            dist = torch.sum(torch.sub(srcFeature, image).pow(2), 1).cpu().numpy() - 1
            if dist < minDist:
                minDist = dist
                index = i

    return minDist, labels[index][0]
