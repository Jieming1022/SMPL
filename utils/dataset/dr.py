
import os
from .imagelist import ImageList


class DR(ImageList):
    """`Diabetic retinopathy dataset for transfer learning.

    Args:
        root (str): Root directory of dataset task (str): The task (domain) to create dataset. Choices include \
                ``'A'``: APTOS2019, ``'D'```: DDR, ``'I'```: IDRiD, ``'M'```: MESSIDOR, ``'M2'```: MESSIDOR2
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a  transformed \
                version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target \
                and transforms it.
    """
    image_list = {
        "A": {
            "original": {
                "train": "APTOS2019/original/train.txt",
                "test": "APTOS2019/original/test.txt",
                "all": "APTOS2019/original/all.txt"
            },
            "cropped": {
                "train": "APTOS2019/cropped/train.txt",
                "test": "APTOS2019/cropped/test.txt",
                "all": "APTOS2019/cropped/all.txt"
            },
            "cropped_enhanced": {
                "train": "APTOS2019/cropped_enhanced/train.txt",
                "test": "APTOS2019/cropped_enhanced/test.txt",
                "all": "APTOS2019/cropped_enhanced/all.txt"
            }
        },
        "D": {
            "original": {
                "train": "DDR/original/train.txt",
                "test": "DDR/original/test.txt",
                "all": "DDR/original/all.txt"
            },
            "cropped": {
                "train": "DDR/cropped/train.txt",
                "test": "DDR/cropped/test.txt",
                "all": "DDR/cropped/all.txt"
            },
            "cropped_enhanced": {
                "train": "DDR/cropped_enhanced/train.txt",
                "test": "DDR/cropped_enhanced/test.txt",
                "all": "DDR/cropped_enhanced/all.txt"
            }
        },
        "E": {
            "original": {
                "train": "EyePACS(Kaggle)/original/train.txt",
                "test": "EyePACS(Kaggle)/original/test.txt",
                "all": "EyePACS(Kaggle)/original/all.txt"
            },
            "cropped": {
                "train": "EyePACS(Kaggle)/cropped/train.txt",
                "test": "EyePACS(Kaggle)/cropped/test.txt",
                "all": "EyePACS(Kaggle)/cropped/all.txt"
            },
            "cropped_enhanced": {
                "train": "EyePACS(Kaggle)/cropped_enhanced/train.txt",
                "test": "EyePACS(Kaggle)/cropped_enhanced/test.txt",
                "all": "EyePACS(Kaggle)/cropped_enhanced/all.txt"
            }
        },
        # TODO: add other datasets
        # "I": {
        #     "train": "IDRiD/train.txt",
        #     "test": "IDRiD/test.txt",
        #     "all": "DDR/all.txt"
        # },
        # "M": {
        #     "train": "Messidor/train.txt",
        #     "test": "Messidor/test.txt",
        #     "all": "DDR/all.txt"
        # },
        # "M2": {
        #     "train": "Messidor-2/train.txt",
        #     "test": "Messidor-2/test.txt",
        #     "all": "DDR/all.txt"
        # },
    }
    DX = {
        "RDR": ['non-referable', 'referable'],
        "NORM": ['normal', 'abnormal'],
        "GRAD": ['normal', 'mild', 'moderate', 'severe', 'proliferative'],
    }

    def __init__(self, root: str, task: str, dx: str, preprocess: str = None, part: str = None, **kwargs):
        assert task in self.image_list
        image_list_path = self.image_list[task][preprocess if preprocess is not None else "cropped"][part if part is not None else "all"]
        data_list_file = os.path.join(root, image_list_path)
        self.classes = DR.DX[dx]

        super(DR, self).__init__(root, self.classes, data_list_file=data_list_file, dx=dx, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())
