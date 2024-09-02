import os
import cv2
from app import FaceAnalysis
from catface_hav_v1.consts import FACE_MODE

""" CONFIG """
NUM_THRESHOLD = 60
DATA_DIR = r"D:\DATA-CNN\Cat-bilibili"
SAVE_DIR = r"D:\DATA-CNN\Catface-embedding"


cnt = 0
def save_single_face(faces):
    global cnt
    dir_path = os.path.join(SAVE_DIR, str(cnt))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # save faces
    for idx, face in enumerate(faces):
        cv2.imwrite(os.path.join(dir_path, f"{idx}.jpg"), face.img)
    cnt += 1


if __name__ == '__main__':
    app = FaceAnalysis(verbose=False)

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith('.mp4'):
            continue
        print(f'-----TARGET {filename}------')

        faces = app.get(os.path.join(DATA_DIR, filename), mode=FACE_MODE.single)

        # filtrate the faces
        face_num = len(faces)
        print(f'🐱 INFO: Get catfaces num {face_num}.')
        if face_num > NUM_THRESHOLD:
            print("RUN filtrate...")
            faces = faces[::2]

        save_single_face(faces)

