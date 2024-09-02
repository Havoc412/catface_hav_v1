import os.path as osp


def ensure_model_available(name, root="./model_zoo/models"):
    dir_path = osp.join(root, name)
    if osp.exists(dir_path):
        return dir_path
    # ...
