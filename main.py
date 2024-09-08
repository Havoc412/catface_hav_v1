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
from catface_hav_v1.utils import merge_breeds

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

    faces = []
    for file_name in sorted_files:
        img_path = os.path.join(DIR_PATH, file_name)
        img = cv2.imread(img_path)

        face = Face()
        face.embedding = app.only_get_embedding(img)
        face.breed = app.only_get_breed(img)

        # breed = app.only_get_breed(img)

        faces.append(face)

    print(len(faces))
    centers = dbscan.filtrate_embeddings(faces, show_pca=True)

    print([_['breed'] for _ in centers])

    cnt_sum = 0
    for center in centers:
        print(center['cnt'])
        cnt_sum += center['cnt']
        print(center['breed']['conf'])
        for i in range(len(center['breed']['conf'])):
            center['breed']['conf'][i] *= center['cnt']
        print(center['breed']['conf'])
    breed = merge_breeds([center['breed'] for center in centers], cnt_sum)

    print(breed)




