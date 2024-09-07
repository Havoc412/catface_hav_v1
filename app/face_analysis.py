import glob
import os.path as osp

from catface_hav_v1.utils import ensure_model_available, ensure_object_type, save_frames_interval
import catface_hav_v1.model_zoo as model_zoo

from catface_hav_v1.align import AT
from catface_hav_v1.structs import Face
from catface_hav_v1.consts import FILE_TYPE, FACE_MODE


class FaceAnalysis:
    _detect = ['obb', 'pose']

    def __init__(self, name="group-1", root="./model_zoo/models", obb_threshold=.5, pose_threshold=.5, **kwargs):
        self.models = {}
        self.model_dir = ensure_model_available(name, root=root)
        # 这里获得的是完整的路径
        pt_files = glob.glob(osp.join(self.model_dir, '*.pt')) + glob.glob(osp.join(self.model_dir, '*.pth'))
        print(pt_files)
        for pt_file in pt_files:
            model = model_zoo.get_model(pt_file, **kwargs)
            if model is None:
                print('Model not recognized:', model)
            else:
                self.models[model.task_name] = model
        self.obb_threshold = obb_threshold
        self.pose_threshold = pose_threshold
        self.at = AT()

    def get(self, target, mode=FACE_MODE.single, only_detect=False, **kwargs):
        """
        之后可以再此模块前增加一些 前置的检测任务。
        :param only_detect: 是否只运行 detect 部分。
        :param mode: 运行模式
        :param target: 支持 cv2 读取后的实体，或者支持的文件类型
        :return:
        """
        assert FACE_MODE.check(mode), "❌ Invalid mode, only [multi, single] are supported."
        # init
        obj, ty = ensure_object_type(target)
        imgs = []
        if ty is FILE_TYPE.img:
            imgs.append(obj)
        elif ty is FILE_TYPE.video:
            imgs = save_frames_interval(obj, **kwargs)
        else:
            print(f"❌ FILE_TYPE [{ty}] of input is not supported.")
            return []
        # detect and other func
        faces = []
        if mode == FACE_MODE.single:
            for img in imgs:
                face = self.detect_img(img)
                if len(face) == 0:
                    continue
                else:
                    face = face[0]
                faces.append(face)
        elif mode == FACE_MODE.multi:
            # todo 实现具体的 多 cat 识别。
            pass

        # other function  # todo 在这里获取 embedding 等。
        if not only_detect:
            for task, model in self.models.items():
                if task in self._detect:
                    continue
                for face in faces:
                    model.get(face)

        return faces

    def detect_img(self, img):
        """
        根据 obb 和 pose 共同来 detect。
        现阶段还是针对一只猫的照片，zor say，the first cat.
        :return: Face[]
        """
        model_obb = self.models['obb']
        model_pose = self.models['pose']

        # stage-1: obb try
        img_obb, box_obb = model_obb.get(img, threshold=self.obb_threshold)
        if img_obb is not None:
            img = img_obb
        del img_obb

        # stage-2: pose
        img_pose, result = model_pose.get(img, box_obb, threshold=self.pose_threshold)
        if result is None:
            return []

        res = []  # todo UPDATE 这里就先写为 list 形式了。
        results = [result]
        for r in results:
            pbox = r.boxes.xyxy[0]
            conf = r.boxes.conf[0]
            kpts = r.keypoints.data[0]

            face = Face(pbox, conf, kpts)
            if box_obb is not None:
                face.change_eye_kpts(box_obb.eye_kpts)  # todo 目前 pose 太差了。

            # tasks
            self.at.get(img_pose, face)

            res.append(face)

        return res

    def only_get_embedding(self, face_img):
        """
        # todo 之后可以更新掉了
        :param face_img: 输入一个 Img
        :return:
        """
        import numpy as np
        assert isinstance(face_img, np.ndarray)
        embedding_model = self.models['embedding']
        embedding = embedding_model.get_for_img(face_img)
        return embedding



