import sys
import torchvision.transforms as T

sys.path.append('../../..')
from .dr import DR
from utils.transforms import ResizeImage


def get_supervised_dataset(root, task, train_transform, val_transform, dx=None, preprocess=None):
    train_dataset = DR(root=root, task=task, part='train', dx=dx, preprocess=preprocess, transform=train_transform)
    val_dataset = DR(root=root, task=task, part='test', dx=dx, preprocess=preprocess, transform=val_transform)
    test_dataset = val_dataset

    class_names = train_dataset.classes
    num_classes = len(class_names)
    return train_dataset, val_dataset, test_dataset, num_classes, class_names


def get_train_transform(
        resizing='default',
        random_horizontal_flip=True,
        random_vertical_flip=False,
        random_rotation=0,
        random_color_jitter=False,
        resize_size=224,
        norm_mean=(0.485, 0.456, 0.406),
        norm_std=(0.229, 0.224, 0.225),
):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224)
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if random_vertical_flip:
        transforms.append(T.RandomVerticalFlip())
    if random_rotation > 0:
        transforms.append(T.RandomRotation(random_rotation))
    if random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
