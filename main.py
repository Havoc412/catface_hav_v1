"""
测试 breed && embedding
"""

import os
import os.path as osp
import cv2
import numpy as np
import pandas as pd

from catface_hav_v1.app import FaceAnalysis
from catface_hav_v1.structs import Face
from catface_hav_v1.utils import dir_str_to_int

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

""" CONFIG """


if __name__ == "__main__":
    app = FaceAnalysis(verbose=False)

    img = cv2.imread(r"D:\DATA-CNN\DATA-CatSingle\爱赖床的图图\00692hmjly1hbv13kc94zj30lw0wh0wt.jpg")

    faces = app.get(img)
    print(len(faces))

    for face in faces:
        print(face.embedding[:3], face.breed)

