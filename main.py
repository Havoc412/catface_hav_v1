"""
from-mp4-dir-2.py  # 本质和第一版相同，主要就是文件读取存放的处理逻辑不同。

## 前置步骤下的简化。
由于 Creeper 中前置考虑了 embedding 文件夹的 id 问题，毕竟保存到 DB 中持久化，
所以只需要照搬文件夹 ID。

从一个 MP4 的文件夹中，遍历 video 文件，隔帧处理，作为一个对象的数据集。
ROOT_DIR
├── cat_1
│   ├── 1.mp4
│   ├── 2.mp4
│   └── ...
└── ...
"""

import os
import cv2

from app import FaceAnalysis
from catface_hav_v1.consts import FACE_MODE
from catface_hav_v1.utils import save_single_faces

""" CONFIG """
NUM_THRESHOLD = 60
DATA_DIR = r"D:\DATA-CNN\Cat-bilibili"
SAVE_DIR = r"D:\DATA-CNN\Catface-embedding"


if __name__ == '__main__':
    app = FaceAnalysis(verbose=False)

    for dir_name in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_name)
        print(f'-----TARGET {dir_name}------')

        faces = []
        for file_name in os.listdir(dir_path):
            if not file_name.endswith('.mp4'):
                continue
            temp_faces = app.get(os.path.join(dir_path, file_name), mode=FACE_MODE.single, only_detect=True)

            faces.extend(temp_faces)

        face_num = len(faces)
        print(f'🐱 INFO: Get catfaces num {face_num}.')
        if face_num > NUM_THRESHOLD:
            print("RUN filtrate...")
            faces = faces[::2]

        save_path = os.path.join(SAVE_DIR, dir_name)
        save_single_faces(save_path, faces)
