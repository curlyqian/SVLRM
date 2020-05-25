from os.path import join
from data_utils import DatasetFromFolder





def get_training_set(dataset, upscale_factor=4, crop=None):
    root_dir = join("dataset", dataset)
    train_dir = join(root_dir, "RGBD_data")

    return DatasetFromFolder(train_dir,
                             upscale_factor=upscale_factor,
                             crop=crop,)


def get_validation_set(dataset, upscale_factor=4):
    root_dir = join("dataset", dataset)
    validation_dir = join(root_dir, "RGBD_testdata")

    return DatasetFromFolder(validation_dir,
                             upscale_factor=upscale_factor)


def get_test_set(dataset, upscale_factor=4):
    root_dir = join("dataset", dataset)
    test_dir = join(root_dir, "RGBD_testdata")

    return DatasetFromFolder(test_dir,
                             upscale_factor=upscale_factor)
