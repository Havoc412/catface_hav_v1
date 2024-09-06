import cv2
from ultralytics import YOLO

from catface_hav_v1.structs import Face


class Breed:
    def __init__(self, model_file=None, verbose=False):
        assert model_file is not None
        self.model_file = model_file
        self.model = YOLO(model_file)

        self.verbose = verbose
        self.task_name = "breed"

        self.names = self.model.names

    def get(self, face):
        assert isinstance(face, Face)
        # cls 任务一定会有一个结果的吧。
        result = self.model(face.img, verbose=self.verbose)[0]
        probs = result.probs.cpu()
        # load res
        breed_list = []
        for ty, conf in zip(probs.top5, probs.top5conf):
            data = {
                'breed': self.names[ty],
                'conf': conf.item()
            }
            breed_list.append(data)
        face.breed = breed_list
        return face.breed


if __name__ == "__main__":
    model = Breed(model_file="./models/group-1/breed.pt")
    # print(model.names)
    face = Face()
    face.img = cv2.imread(r"D:\DATA-CNN\Catface-embedding\415\2.jpg")
    print(model.get(face))



