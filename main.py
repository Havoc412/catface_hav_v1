"""
实现 norm && cal_similarity，以及相关测试。
ROOT_DIR
├── 0.jpg
├── 1.jpg
├── 2.jpg
└── ...
"""

import os
import os.path as osp
import cv2
import numpy as np
import pandas as pd

from catface_hav_v1.app import FaceAnalysis
from catface_hav_v1.structs import Face

""" CONFIG """
BASE_DIR = "./test/data/faces-single"


if __name__ == "__main__":
    app = FaceAnalysis(verbose=False)

    # get embedding
    embeddings = []
    labels = os.listdir(BASE_DIR)
    for img_name in labels:
        img_path = os.path.join(BASE_DIR, img_name)

        img = cv2.imread(img_path)
        face = Face()
        face.img = img

        app.get_embedding(face)
        embeddings.append(face.normed_embedding)

    # cal sim
    # dot_list = [np.dot(i, j) for i in embeddings for j in embeddings]
    # frame = pd.DataFrame(np.array(dot_list).reshape(len(embeddings), len(embeddings)),
    #                      index=labels, columns=labels)
    # print(frame)

    # cal && sum
    dot_sum_list = []
    for i in embeddings:
        dot_sum = 0
        for j in embeddings:
            dot_sum += np.dot(i, j)
        dot_sum_list.append(dot_sum)

    for i, dot_sum in zip(labels, dot_sum_list):
        print(i, dot_sum)





