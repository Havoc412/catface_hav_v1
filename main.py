"""
测试 dbscan
"""

import os
import os.path as osp
import cv2
import numpy as np
import pandas as pd

from catface_hav_v1.app import FaceAnalysis, DBSCAN
from catface_hav_v1.structs import Face
from catface_hav_v1.utils import dir_str_to_int

from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

""" CONFIG """
DIR_PATH = r"D:\DATA-CNN\Catface-embedding\69"


if __name__ == "__main__":
    # 列出目录中的文件
    files = os.listdir(DIR_PATH)
    # 按文件名数字排序
    sorted_files = sorted(files, key=lambda x: int(x.split('.')[0]))

    app = FaceAnalysis(verbose=False)
    dbscan = DBSCAN(eps=.4, verbose=True)

    embeddings = []
    for file_name in sorted_files:
        img_path = os.path.join(DIR_PATH, file_name)
        img = cv2.imread(img_path)
        embedding = app.only_get_embedding(img)

        breed = app.only_get_breed(img)
        print(breed)
        exit(0)

        embeddings.append(embedding)

    print(len(embeddings))
    centers = dbscan.filtrate_embeddings(embeddings, show_pca=True)
    print(len(centers))




