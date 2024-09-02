import os.path as osp

from .obb import Obb
from .pose import Pose


class ModelRouter:
    def __init__(self, pt_file):
        self.pt_file = pt_file

    def get_model(self, **kwargs):
        #  todo 之后这里区分一下 yolo 和 embedding 的模型
        if self.pt_file.endswith("embedding.pt"):
            return None  # todo wait for embedding part
        elif self.pt_file.endswith("obb.pt"):
            verbose = kwargs.get("verbose", False)
            return Obb(self.pt_file, verbose)
        elif self.pt_file.endswith("pose.pt"):
            verbose = kwargs.get("verbose", False)
            return Pose(self.pt_file, verbose)
        else:
            print(f"⚠️ Could not support model [{self.pt_file}], only for obb, pose, embedding.")
            return None


def get_model(path, **kwargs):
    assert path.endswith('.pt'), f"[{path}] is not a pt file."
    assert osp.exists(path), f"[{path}] is not exists."
    assert osp.isfile(path), f"[{path}] is not a file, please check."
    router = ModelRouter(path)
    model = router.get_model(**kwargs)
    return model
