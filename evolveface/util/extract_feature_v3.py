# Helper function for extracting features from pre-trained models
import torch
import cv2
import numpy as np
import os
from evolveface.util.perf_utils import lp_wrapper
from cvtorchvision import cvtransforms
from torch.autograd import Variable


def l2_norm(tensor, axis=1):
    norm = torch.norm(tensor, 2, axis, True)
    output = torch.div(tensor, norm)
    return output


def extract_feature(img_root, backbone,
                    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta=True, original=1):
    # prerequisites
    if original == 1:
        assert (os.path.exists(img_root))
        print('Testing Data Root:', img_root)

    # load image
    if original == 1:
        mat = cv2.imread(img_root)
    else:
        mat = img_root

    resized = cv2.cvtColor(cv2.resize(mat, (112, 112)), cv2.COLOR_BGR2RGB)

    # load numpy to tensor
    ccropped = resized.swapaxes(1, 2).swapaxes(0, 1)
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype=np.float32)
    ccropped = (ccropped - 127.5) / 128.0
    flipped = np.flip(ccropped)
    ccropped = torch.from_numpy(ccropped)
    flipped = torch.from_numpy(np.ascontiguousarray(flipped))

    with torch.no_grad():
        if tta:
            emb_batch = backbone(ccropped.to(device)).cpu() + backbone(flipped.to(device)).cpu()
            features = l2_norm(emb_batch)
        else:
            features = l2_norm(backbone(ccropped.to(device)).cpu())
    return features


def get_embeddings(image, net, device):
    transform = cvtransforms.Compose(
        [cvtransforms.Resize((112, 112)), cvtransforms.RandomHorizontalFlip(), cvtransforms.ToTensor(),
         cvtransforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transformed_image = transform(image).to(device)
    the_image = Variable(transformed_image).unsqueeze(0)
    # net.eval()
    embeddings = l2_norm(net.forward(the_image)).detach()   # remain data at gpu
    return embeddings
