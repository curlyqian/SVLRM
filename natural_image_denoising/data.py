from os.path import join
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from data_utils import DatasetFromFolder




def get_training_set(dataset, add_noise=None, convert_L=None,crop=None):
    root_dir = join("dataset", dataset)
    train_dir = join(root_dir, "train")


    return DatasetFromFolder(train_dir,
                             add_noise=add_noise,
                             convert_L=convert_L,
                             crop=crop)


def get_validation_set(dataset, add_noise=None):
    root_dir = join("dataset", dataset)
    validation_dir = join(root_dir, "valid")

    return DatasetFromFolder(validation_dir,
                             add_noise=add_noise)
'''
def get_test_set(dataset, add_noise=None):
    root_dir = join("dataset", dataset)
    validation_dir = join(root_dir, "valid")

    return DatasetFromFolder(validation_dir,
                             add_noise=add_noise)
'''

def get_test_set(dataset):
    root_dir = join("dataset", dataset)
    test_dir = join(root_dir, "denoising_inputs")

    return DatasetFromFolder(test_dir)
