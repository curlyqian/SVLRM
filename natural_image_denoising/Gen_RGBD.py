import os
import glob
import os.path as osp
import cv2
import numpy as np



def load_img(file_path,type):
    dir_path = os.path.join(os.getcwd(), file_path)
    img_path = glob.glob(os.path.join(dir_path, type))
    return img_path

def read_img(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
    return image




def save(sub_ip, savepath = 'data'):
    filepath = savepath+'/'
    for i in range(len(sub_ip)):
        cv2.imwrite(filepath + np.str(i) + '.jpg', sub_ip[i],[int(cv2.IMWRITE_JPEG_QUALITY),100])


def data_aug(rgb_path = '', depth_path = '', savepath = 'RGBD_data'):
    rgb_path = load_img(rgb_path,'*.bmp')
    depth_path = load_img(depth_path, '*.bmp')
    print(rgb_path)
    print(depth_path)
    save_path = savepath
    if not osp.exists(save_path):
        os.makedirs(save_path)
    for i in range(len(rgb_path)):
        rgb_image = read_img(rgb_path[i])
        depth_image = read_img(depth_path[i])
        rgb = np.array(rgb_image)
        depth = np.array(depth_image)
        rgbd = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
        rgbd[:, :, 0] = rgb[:, :, 0]
        rgbd[:, :, 1] = rgb[:, :, 1]
        rgbd[:, :, 2] = rgb[:, :, 2]
        rgbd[:, :, 3] = depth
        print('data no.',i)
        filepath = savepath + '/'
        cv2.imwrite(filepath + np.str(i) + '.png',rgbd)
        print('---------save---------')

if __name__ == '__main__':
    print('starting data augmentation...')
    rgb_path = 'E:/BaiduNetdiskDownload/depth_denoising_test_data_20200521_105703/depth_denoising_test_data/input_rgb'
    depth_path = 'E:/BaiduNetdiskDownload/depth_denoising_test_data_20200521_105703/depth_denoising_test_data/noise_input'
    #rgb_path = 'images'
    #depth_path = 'depth_gt'
    savepath = 'RGBD_testdata'
    data_aug(rgb_path,depth_path,savepath)
