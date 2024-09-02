import os

import cv2
from cv2 import VideoCapture as Video
from numpy import ndarray as Img

from catface_hav_v1.consts import FILE_TYPE


def ensure_object_type(obj):
    """
    确保 FaceAnalysis 处理的是一个 cv2 的实体，同时判断出实体的类型。 Video / Image.
    :param obj:
    :return:
    """
    if isinstance(obj, Video):
        return obj, FILE_TYPE.img
    elif isinstance(obj, Img):
        return obj, FILE_TYPE.video
    elif os.path.exists(obj) and os.path.isfile(obj):
        file_path = obj
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext in ['.jpeg', '.jpg', '.png', '.bmp', '.tiff', '.webp']:  # 图片文件
            # 使用cv2.imread()来读取图片
            image = cv2.imread(obj)
            if image is not None:
                return image, FILE_TYPE.img
            else:
                print(f"❌ Error: Couldn't open the file [{file_path}] by cv2.imread.")
        elif ext in ['.avi', '.mp4', '.mov', '.mkv', '.flv', '.wmv']:  # 视频文件
            # 使用cv2.VideoCapture()来读取视频
            video = cv2.VideoCapture(file_path)
            if video.isOpened():
                return video, FILE_TYPE.video
            else:
                print(f"❌ Error: Couldn't open the file [{file_path}] by cv2.VideoCapture.")
        else:
            print("❌ Not a supported file type.")
    else:
        print("❌ Not a cv2 object or available file path.")
    return [None] * 2


def save_frames_interval(video, **kwargs):
    """
    从 Video 对象中隔x帧保存 image
    :param video:
    :return: 返回 imgs，但实际上 list.append 函数会直接作用于原list，所以不是很必须。
    """
    if not video.isOpened():
        return []
    interval = kwargs.get('video_interval', 1)
    target_num = kwargs.get('video_num', 30)

    # FRAME-CNT and FPS
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_interval = interval * fps
    if frame_count // frame_interval > target_num:
        frame_interval = frame_count // target_num // 4  # 考虑选取的一帧可能无效 cat face

    frames = []
    current_frame = 0
    while current_frame < frame_count:
        video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
        current_frame += frame_interval
    video.release()
    print(f"📼 INFO: Video get {len(frames)} frames.")
    return frames


if __name__ == "__main__":
    mp4 = cv2.VideoCapture(r"D:\DATA-CNN\CAT-MP4\ex3.mp4")
    print(ensure_object_type(mp4))
