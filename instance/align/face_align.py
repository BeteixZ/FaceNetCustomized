import argparse
import os

import numpy
from PIL import Image
from tqdm import tqdm

from evolveface.align.align_trans import warp_and_crop_face, get_reference_facial_points
from evolveface.align.detector import detect_faces

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face Alignment & Resize")
    parser.add_argument("-src_root", "--src_root", help="specify your source dir", default="../database/Alpha/",
                        type=str)
    parser.add_argument("-dst_root", "--dst_root", help="specify your destination dir", type=str,
                        default="../database/AlphaAligned/")
    parser.add_argument("-crop_size", "--crop_size", help="specify size of aligned faces, align and crop with "
                                                          "padding, default is 112",
                        default=112, type=int)
    args = parser.parse_args()

    src_root = args.src_root
    dst_root = args.dst_root
    crop_size = args.crop_size
    scale = crop_size / 112.0
    reference = get_reference_facial_points(default_square=True) * scale

    # cwd = os.getcwd()
    # os.chdir(src_root)
    # os.system("find . -name '*.DS_Store' -type f -delete")  # only in MacOS
    # os.chdir(cwd)

    if not os.path.isdir(dst_root):
        os.mkdir(dst_root)

    for subFolder in tqdm(os.listdir(src_root)):
        if not os.path.isdir(os.path.join(dst_root, subFolder)):
            os.mkdir(os.path.join(dst_root, subFolder))  # create duplicate folders at dst_root
        for image in os.listdir(os.path.join(src_root, subFolder)):
            print("Processing\t{}".format(os.path.join(subFolder, image)))
            img = Image.open(os.path.join(src_root, subFolder, image))
            try:
                boundingBoxes, landmarks = detect_faces(img)
            except Exception:
                print("{} is discarded due to exception!".format(os.path.join(src_root, subFolder, image)))
                continue
            if len(landmarks) == 0:
                print("{} is discarded due to non-detected landmarks!".format(
                    os.path.join(src_root, subFolder, image)))
                continue
            facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
            warped_face = warp_and_crop_face(numpy.array(img), facial5points, reference,
                                             crop_size=(crop_size, crop_size))
            img_warped = Image.fromarray(warped_face)
            if image.split('.')[-1].lower() not in ['jpg', 'jpeg']:  # not from jpg
                image = '.'.join(image.split('.')[:-1]) + '.jpg'
            img_warped.save(os.path.join(dst_root, subFolder, image))
