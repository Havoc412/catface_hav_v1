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
        face.breed = self.get_for_img(face.img)
        return face.breed

    def get_for_img(self, img):
        result = self.model(img, verbose=self.verbose)[0]
        probs = result.probs.cpu()
        # load res
        breed = {
            'top5': [self.names[i] for i in probs.top5],
            'conf': probs.top5conf.tolist()
        }
        return breed


if __name__ == "__main__":
    model = Breed(model_file="./models/group-1/breed.pt")
    # print(model.names)
    face = Face()
    face.img = cv2.imread(r"D:\DATA-CNN\Catface-embedding\415\2.jpg")
    print(model.get(face))
    print(face.breed)



