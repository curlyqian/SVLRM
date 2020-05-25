import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import RandomCrop, ToTensor, Resize, ToPILImage, RandomHorizontalFlip, \
    RandomVerticalFlip, RandomRotation


'''
def downsampling(img,upscale_factors):
    img_array = np.asarray(img)
    height = img_array.shape[0]
    width = img_array.shape[1]
    lr_height,lr_width = height//upscale_factors,width//upscale_factors
    new_image = np.zeros((lr_height,lr_width),np.uint8)
    for i in range(lr_height):
        x=i*upscale_factors
        for j in range(lr_width):
            y=j*upscale_factors
            new_image[i][j]=img_array[x][y]
    lr_image = Image.fromarray(new_image)
    return lr_image
'''
def downsampling(img,upscale_factors):
    img_array = np.asarray(img)
    lr_image = img_array[upscale_factors-1::upscale_factors,upscale_factors-1::upscale_factors]
    lr_image = Image.fromarray(lr_image)

    return lr_image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGBA')
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, upscale_factor=4, crop=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x)
                                for x in sorted(listdir(image_dir)) if is_image_file(x)]  #文件名列表

        self.crop = crop
        self.upscale_factor = upscale_factor

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])#input是预先合成的4通道RGDB图片
        #数据增强
        if self.crop:
            input = RandomCrop(64)(input)#取patch
            input = RandomHorizontalFlip()(input)#水平翻转
            input = RandomVerticalFlip()(input)#竖直翻转
            input = RandomRotation(180)(input)#随机旋转
        input_tensor = ToTensor()(input)
        rgb_tensor = torch.zeros(3,input_tensor.shape[1],input_tensor.shape[2])
        depth_tensor = torch.zeros(1, input_tensor.shape[1], input_tensor.shape[2])
        rgb_tensor[0, :, :] = input_tensor[0, :, :]
        rgb_tensor[1, :, :] = input_tensor[1, :, :]
        rgb_tensor[2, :, :] = input_tensor[2, :, :]
        depth_tensor[0, :, :] = input_tensor[3, :, :]
        depth = ToPILImage()(depth_tensor)
        size = min(depth.size[0], depth.size[1])
        guide = ToPILImage()(rgb_tensor)
        target = depth.copy()

        guide = guide.convert('L')
        #生成LR
        depth = downsampling(depth,self.upscale_factor)
        depth = Resize(size=size,interpolation=Image.BICUBIC)(depth)

        depth = ToTensor()(depth)
        guide = ToTensor()(guide)
        depth = torch.cat((depth,guide),0)#concatenate 生成输入张量
        target = ToTensor()(target)

        return depth, target

    def __len__(self):
        return len(self.image_filenames)