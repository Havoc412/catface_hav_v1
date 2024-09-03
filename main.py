"""
接下来。
是 Embedding 的整合。
需要实现的效果是：根据这样的文件结构，计算 Embeeding，然后打上 label 输入到 torchboard 来查看。
ROOT_DIR
├── cat_1
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── cat_2
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── ...
"""
import os
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter


from app import FaceAnalysis
from catface_hav_v1.structs import Face

""" CONFIG """
TAR_DIR = r"D:\DATA-CNN\Embedding-test"
embedding_dim = 512


if __name__ == '__main__':
    app = FaceAnalysis(verbose=False)

    #  cal all embedding
    embeddings = []
    labels = []
    for cat_name in os.listdir(TAR_DIR):
        dir_path = os.path.join(TAR_DIR, cat_name)
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)

            # 特化使用 Face 类，直接导入 obb-pose-at 处理完后的 img。
            img = cv2.imread(img_path)
            face = Face()
            face.img = img

            embedding = app.get_embedding(face)
            embeddings.append(embedding)
            labels.append(cat_name)

    print(len(embeddings), len(labels))
    # write to
    embeddings = np.array(embeddings)
    writer = SummaryWriter('runs/catface-1')
    writer.add_embedding(embeddings, metadata=labels, tag="catface")

    writer.close()


















