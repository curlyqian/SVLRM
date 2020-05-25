import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import RandomCrop, ToTensor, ToPILImage, RandomHorizontalFlip, \
    RandomVerticalFlip, RandomRotation



def noisy(img):
    if img.mode=='L':
        gauss = np.random.normal(size=(img.height, img.width)) * 0.01 * np.random.randint(1, 10) * 255
    else:
        gauss = np.random.normal(size=(img.height, img.width, 3)) * 0.01 * np.random.randint(1, 10) * 255
    noisy = np.uint8(np.clip(img + gauss, 0, 255))
    return noisy



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath)
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, add_noise=None, convert_L=None,crop=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x)
                                for x in listdir(image_dir) if is_image_file(x)]  #文件名列表

        self.add_noise = add_noise
        self.convert_L = convert_L
        self.crop = crop

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        #加数据增强
        if self.crop:
            input = RandomCrop(64)(input)
            input = RandomHorizontalFlip()(input)
            input = RandomVerticalFlip()(input)
            input = RandomRotation(180)(input)
        target = input.copy()
        if self.convert_L:
            input = input.convert('L')
            target = target.convert('L')
        if self.add_noise:
            input = noisy(input)
            input = ToPILImage()(input)
        input = ToTensor()(input)
        target = ToTensor()(target)
        guide = input.clone()
        input = torch.cat((input,guide),0)

        return input, target

    def __len__(self):
        return len(self.image_filenames)