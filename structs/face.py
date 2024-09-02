import numpy as np


class Face:
    def __init__(self, box, conf, kpts):
        self.box = box
        self.conf = conf
        self.kpts = kpts.numpy()[:, :2]

        # self.img  # by detect

    def change_eye_kpts(self, eye_kpts):
        kpts = np.array(eye_kpts)
        # handle rotate 180°
        if kpts[0, 0] > kpts[1, 0]:
            kpts[[0, 1]] = kpts[[1, 0]]  # tip 简洁写法
        self.kpts[:2] = kpts
