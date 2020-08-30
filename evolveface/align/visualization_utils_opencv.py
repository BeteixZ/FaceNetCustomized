import cv2 as cv


def show_result(mat, boundingBoxes, facialLandmarks=[]):
    """
    Draw bounding boxes and facial landmarks.
    :param mat: an instance of cv matrix.
    :param boundingBoxes: a float numpy array of shape [n, 5].
    :param facialLandmarks: a float numpy array of shape [n, 10].
    :return: an matrix with bounding boxes & landmarks.
    """
    mat_copy = mat
    for box in boundingBoxes:
        cv.rectangle(mat_copy, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), 2)

    for pt in facialLandmarks:
        for i in range(5):
            cv.circle(mat, (int(pt[i]), int(pt[i + 5])), 2, (255, 0, 0), 1)

    return mat_copy
